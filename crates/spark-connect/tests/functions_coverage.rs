//! Exercise the newer Spark 4.2 SQL function wrappers. Each builds a `Column` by
//! constructing an expression (no server needed), so calling them pins their bodies.

use spark_connect::functions as f;
use spark_connect::functions::col;

#[test]
fn newer_function_builders() {
    let c = || col("a");
    let d = || col("b");

    // time / timestamp constructors and converters
    let _ = f::make_time(c(), c(), c());
    let _ = f::make_timestamp(c(), c(), c(), c(), c(), c());
    let _ = f::make_timestamp_ltz(c(), c(), c(), c(), c(), c());
    let _ = f::make_timestamp_ntz(c(), c(), c(), c(), c(), c());
    let _ = f::try_make_timestamp(c(), c(), c(), c(), c(), c());
    let _ = f::try_make_timestamp_ltz(c(), c(), c(), c(), c(), c());
    let _ = f::try_make_timestamp_ntz(c(), c(), c(), c(), c(), c());
    let _ = f::to_time(c());
    let _ = f::try_to_time(c());
    let _ = f::try_to_date(c());
    let _ = f::timestamp_nanos(c());
    let _ = f::unix_nanos(c());
    let _ = f::time_bucket(c(), d());
    let _ = f::time_diff(c(), c(), d());
    let _ = f::time_from_micros(c());
    let _ = f::time_from_millis(c());
    let _ = f::time_from_seconds(c());
    let _ = f::time_to_micros(c());
    let _ = f::time_to_millis(c());
    let _ = f::time_to_seconds(c());
    let _ = f::time_trunc(c(), d());
    let _ = f::current_time();
    let _ = f::current_path();

    // geospatial
    let _ = f::st_asbinary(c());
    let _ = f::st_geogfromwkb(c());
    let _ = f::st_geomfromwkb(c());
    let _ = f::st_setsrid(c(), d());
    let _ = f::st_srid(c());

    // theta sketches
    let _ = f::theta_difference(c(), d());
    let _ = f::theta_intersection(c(), d());
    let _ = f::theta_intersection_agg(c());
    let _ = f::theta_sketch_agg(c());
    let _ = f::theta_sketch_estimate(c());
    let _ = f::theta_union(c(), d());
    let _ = f::theta_union_agg(c());

    // KLL sketches (bigint/double/float families)
    let _ = f::kll_merge_agg_bigint(c());
    let _ = f::kll_merge_agg_double(c());
    let _ = f::kll_merge_agg_float(c());
    let _ = f::kll_sketch_agg_bigint(c());
    let _ = f::kll_sketch_agg_double(c());
    let _ = f::kll_sketch_agg_float(c());
    let _ = f::kll_sketch_get_n_bigint(c());
    let _ = f::kll_sketch_get_n_double(c());
    let _ = f::kll_sketch_get_n_float(c());
    let _ = f::kll_sketch_get_quantile_bigint(c(), d());
    let _ = f::kll_sketch_get_quantile_double(c(), d());
    let _ = f::kll_sketch_get_quantile_float(c(), d());
    let _ = f::kll_sketch_get_rank_bigint(c(), d());
    let _ = f::kll_sketch_get_rank_double(c(), d());
    let _ = f::kll_sketch_get_rank_float(c(), d());
    let _ = f::kll_sketch_merge_bigint(c(), d());
    let _ = f::kll_sketch_merge_double(c(), d());
    let _ = f::kll_sketch_merge_float(c(), d());
    let _ = f::kll_sketch_to_string_bigint(c());
    let _ = f::kll_sketch_to_string_double(c());
    let _ = f::kll_sketch_to_string_float(c());

    // variant
    let _ = f::is_valid_variant(c());
    let _ = f::variant_array_append(c(), c(), c());
    let _ = f::variant_delete(c(), vec![c()]);
    let _ = f::variant_insert(c(), c(), c());
    let _ = f::variant_set(c(), c(), c());
    let _ = f::try_variant_array_append(c(), c(), c());
    let _ = f::try_variant_insert(c(), c(), c());
    let _ = f::try_variant_set(c(), c(), c());

    // vector
    let _ = f::vector_avg(c());
    let _ = f::vector_cosine_similarity(c(), d());
    let _ = f::vector_inner_product(c(), d());
    let _ = f::vector_l2_distance(c(), d());

    // misc scalar/agg
    let _ = f::bitmap_and_agg(c());
    let _ = f::chr(c());
    let _ = f::counter_diff(c());
    let _ = f::hmac(c(), d());
    let _ = f::jaro_winkler_similarity(c(), d());
    let _ = f::quote(c());
    let _ = f::sha2(c(), 256);
    let _ = f::uuid();

    // avro / protobuf
    let _ = f::from_avro(c(), "{}");
    let _ = f::from_avro_with_options(c(), "{}", d());
    let _ = f::to_avro(c());
    let _ = f::to_avro_with_schema(c(), "{}");
    let _ = f::from_protobuf(c(), "M");
    let _ = f::from_protobuf_with_options(c(), "M", d());
    let _ = f::to_protobuf(c(), "M");
    let _ = f::to_protobuf_with_options(c(), "M", d());

    // higher-order (closures take the lambda's bound variables as a Vec<Column>)
    let _ = f::map_zip_with(c(), d(), |args| args[0].clone());
    let _ = f::reduce(c(), col("init"), |args| args[0].clone());
}
