//! The spatial reference systems that the spatial types name by their SRID, as Spark 4.3.0 knows
//! them: `SpatialReferenceSystemCache` of `sql/api`, which loads the registry that Spark generates
//! from PROJ 9.8.1 (`dev/generate_srs_registry.py` in apache/spark) and adds its own entries. The
//! registry is a copy of the file that Spark ships as `srs_registry.csv` (see its header); both
//! name the same SRIDs by the same string IDs, so the JSON of a spatial type reads on either side.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The registry of Spark: `srid,string_id,is_geographic` rows, after `#` comment lines.
const REGISTRY_CSV: &str = include_str!("srs_registry.csv");

/// A spatial reference system of the registry.
#[derive(Clone, Copy)]
struct Srs {
    /// The string ID that Spark writes for it, the OGC one where Spark has one.
    string_id: &'static str,
    /// Whether it is a geographic coordinate system: a Geography takes only one of those.
    geographic: bool,
}

/// The lookups of the registry, by SRID and by string ID.
struct Registry {
    by_srid: HashMap<i32, Srs>,
    /// The SRID of each string ID that Spark reads: the OGC IDs and their EPSG aliases included.
    by_string_id: HashMap<&'static str, i32>,
}

impl Registry {
    fn load() -> Self {
        let mut by_srid = HashMap::new();
        let mut by_string_id = HashMap::new();
        for line in REGISTRY_CSV.lines() {
            if line.starts_with('#') || line.starts_with("srid,") {
                continue;
            }
            let mut parts = line.splitn(3, ',');
            let (Some(srid), Some(string_id), Some(geographic)) =
                (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            let Ok(srid) = srid.trim().parse::<i32>() else {
                continue;
            };
            let string_id = string_id.trim();
            let geographic = geographic.trim().eq_ignore_ascii_case("true");
            by_srid.insert(
                srid,
                Srs {
                    string_id,
                    geographic,
                },
            );
            by_string_id.insert(string_id, srid);
        }
        // The entries that Spark adds to the registry: SRID 0 is a Cartesian system with no SRS.
        by_srid.insert(
            0,
            Srs {
                string_id: "SRID:0",
                geographic: false,
            },
        );
        by_string_id.insert("SRID:0", 0);
        // The OGC standardizations take over the string IDs of their EPSG systems, which stay
        // as aliases in `by_string_id`.
        for (srid, ogc) in [
            (4326, "OGC:CRS84"),
            (4267, "OGC:CRS27"),
            (4269, "OGC:CRS83"),
        ] {
            if let Some(existing) = by_srid.get(&srid).copied() {
                by_srid.insert(
                    srid,
                    Srs {
                        string_id: ogc,
                        geographic: existing.geographic,
                    },
                );
                by_string_id.insert(ogc, srid);
            }
        }
        Self {
            by_srid,
            by_string_id,
        }
    }
}

/// The registry, loaded once on first use.
fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(Registry::load)
}

/// Whether a spatial type of the given kind takes the SRID `srid`: a Geography takes only a
/// geographic one, and a Geometry any one of the registry.
pub(crate) fn is_supported(srid: i32, geographic: bool) -> bool {
    registry()
        .by_srid
        .get(&srid)
        .is_some_and(|srs| srs.geographic || !geographic)
}

/// The string ID that Spark writes for the SRID `srid`, if a spatial type of the given kind takes
/// it.
pub(crate) fn string_id(srid: i32, geographic: bool) -> Option<&'static str> {
    registry()
        .by_srid
        .get(&srid)
        .filter(|srs| srs.geographic || !geographic)
        .map(|srs| srs.string_id)
}

/// The SRID of the string ID `crs` that Spark reads, aliases included, if a spatial type of the
/// given kind takes it. The match is exact, as Spark's.
pub(crate) fn srid(crs: &str, geographic: bool) -> Option<i32> {
    let srid = *registry().by_string_id.get(crs)?;
    is_supported(srid, geographic).then_some(srid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_ids_are_sparks() {
        // The OGC IDs of the EPSG systems that Spark overrides, and their EPSG aliases.
        assert_eq!(string_id(4326, false), Some("OGC:CRS84"));
        assert_eq!(string_id(4269, true), Some("OGC:CRS83"));
        assert_eq!(string_id(4267, true), Some("OGC:CRS27"));
        assert_eq!(srid("EPSG:4269", true), Some(4269));
        assert_eq!(srid("EPSG:4326", true), Some(4326));
        // An EPSG system of the registry, and SRID 0.
        assert_eq!(string_id(2154, false), Some("EPSG:2154"));
        assert_eq!(string_id(0, false), Some("SRID:0"));
        assert_eq!(srid("SRID:0", false), Some(0));
    }

    #[test]
    fn a_geography_takes_only_a_geographic_system() {
        assert!(is_supported(4326, true));
        assert!(!is_supported(3857, true));
        assert!(!is_supported(0, true));
        assert!(is_supported(3857, false));
        assert!(is_supported(2154, false));
        assert_eq!(string_id(3857, true), None);
        assert_eq!(srid("EPSG:3857", true), None);
        assert_eq!(srid("EPSG:3857", false), Some(3857));
    }

    #[test]
    fn unknown_srids_and_crs_strings_are_not_supported() {
        assert!(!is_supported(-1, false));
        assert!(!is_supported(99_999_999, false));
        assert_eq!(string_id(99_999_999, false), None);
        // The lookups by string ID are exact: the case and the `SRID:` form of a number are not
        // names of Spark.
        assert_eq!(srid("srid:0", false), None);
        assert_eq!(srid("ogc:crs84", false), None);
        assert_eq!(srid("SRID:4326", false), None);
    }
}
