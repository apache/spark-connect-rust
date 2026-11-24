// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use uuid::Uuid;

/// Matches `UUID.fromString` on OpenJDK 21, including its
/// abbreviated groups, optional plus signs and truncation of oversized fields.
pub(crate) fn parse_uuid(value: &str) -> Option<Uuid> {
    if value.encode_utf16().take(37).count() > 36 {
        return None;
    }
    let mut groups = value.split('-');
    let mut fields = [0_i64; 5];
    for field in &mut fields {
        let group = groups.next()?;
        let digits = group.strip_prefix('+').unwrap_or(group);
        if digits.is_empty() {
            return None;
        }
        *field = digits.chars().try_fold(0_i64, |value, digit| {
            value
                .checked_mul(16)?
                .checked_add(i64::from(hex_digit(digit)?))
        })?;
    }
    if groups.next().is_some() {
        return None;
    }
    Some(Uuid::from_u128(
        ((fields[0] as u128 & 0xffff_ffff) << 96)
            | ((fields[1] as u128 & 0xffff) << 80)
            | ((fields[2] as u128 & 0xffff) << 64)
            | ((fields[3] as u128 & 0xffff) << 48)
            | (fields[4] as u128 & 0xffff_ffff_ffff),
    ))
}

fn hex_digit(digit: char) -> Option<u32> {
    if let Some(value) = digit.to_digit(16) {
        return Some(value);
    }
    match digit {
        '\u{ff21}'..='\u{ff26}' => Some(digit as u32 - 0xff21 + 10),
        '\u{ff41}'..='\u{ff46}' => Some(digit as u32 - 0xff41 + 10),
        _ => {
            // BMP decimal-zero code points accepted by Character.digit(char, 16)
            // in OpenJDK 21 (Unicode 15.0). Java parses UTF-16 chars here, so
            // supplementary-plane digits are deliberately excluded.
            const ZEROES: &[u32] = &[
                0x0660, 0x06f0, 0x07c0, 0x0966, 0x09e6, 0x0a66, 0x0ae6, 0x0b66, 0x0be6, 0x0c66,
                0x0ce6, 0x0d66, 0x0de6, 0x0e50, 0x0ed0, 0x0f20, 0x1040, 0x1090, 0x17e0, 0x1810,
                0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0,
                0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10,
            ];
            ZEROES
                .iter()
                .find_map(|&zero| (digit as u32).checked_sub(zero).filter(|&n| n < 10))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_uuid;

    #[test]
    fn accepts_jdk_uuid_spellings() {
        for value in [
            "1-2-3-4-5",
            "+1-+2-+3-+4-+5",
            "00000000000000001-2-3-4-5",
            "0000000000000000000000000001-2-3-4-5",
            "100000001-10002-3-4-5",
            "１-２-３-４-５",
            "١-٢-٣-٤-٥",
        ] {
            assert_eq!(
                parse_uuid(value).unwrap().to_string(),
                "00000001-0002-0003-0004-000000000005",
                "{value}"
            );
        }
        assert_eq!(
            parse_uuid("123E4567-E89B-12D3-A456-426614174000")
                .unwrap()
                .to_string(),
            "123e4567-e89b-12d3-a456-426614174000"
        );
        assert_eq!(
            parse_uuid("Ａ-Ｂ-Ｃ-Ｄ-Ｅ").unwrap().to_string(),
            "0000000a-000b-000c-000d-00000000000e"
        );
    }

    #[test]
    fn masks_each_field_after_parsing_a_signed_long() {
        for (value, expected) in [
            (
                "7fffffffffffffff-2-3-4-5",
                "ffffffff-0002-0003-0004-000000000005",
            ),
            (
                "1-10002-10003-10004-5",
                "00000001-0002-0003-0004-000000000005",
            ),
            (
                "1-2-3-4-7fffffffffffffff",
                "00000001-0002-0003-0004-ffffffffffff",
            ),
        ] {
            assert_eq!(parse_uuid(value).unwrap().to_string(), expected);
        }
    }

    #[test]
    fn rejects_forms_the_jdk_does_not_parse() {
        for value in [
            "",
            "1-2-3-4",
            "1-2-3-4-5-6",
            "-1-2-3-4-5",
            "1--2-3-4-5",
            "1-2-3-4-",
            "1-2-3-4-+",
            "1-2-3-4-++5",
            "1-2-3-4- 5",
            "1-2-3-4-0x5",
            "8000000000000000-2-3-4-5",
            "1-2-3-4-8000000000000000",
            "00000000000000000000000000001-2-3-4-5",
            "123e4567e89b12d3a456426614174000",
            "{123e4567-e89b-12d3-a456-426614174000}",
            "urn:uuid:123e4567-e89b-12d3-a456-426614174000",
            "1-2-3-4-⑤",
            "1-2-3-4-𝟝",
        ] {
            assert!(parse_uuid(value).is_none(), "{value}");
        }
    }
}
