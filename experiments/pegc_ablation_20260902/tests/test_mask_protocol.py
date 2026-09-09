import unittest

from evaluation.mask_protocol import (
    cyclegrpo_legacy_raw_mask_token_count,
    complete_mask_group_count,
    parse_cyclegrpo_legacy_mask_groups,
    parse_mask_groups,
    validate_mask_protocol,
)


class MaskProtocolTest(unittest.TestCase):
    def setUp(self):
        self.response = (
            "<|mt_start|><|mt_0007|><|mt_0268|><|mt_end|>"
            "<|mt_start|><|mt_0008|><|mt_0269|><|mt_end|>"
        )

    def test_legacy_union_keeps_all_legal_groups(self):
        self.assertEqual(
            parse_mask_groups(self.response, codebook_size=256, protocol="legacy_union"),
            [[7, 12], [8, 13]],
        )

    def test_first_mask_keeps_only_the_first_complete_group(self):
        self.assertEqual(
            parse_mask_groups(self.response, codebook_size=256, protocol="first_mask"),
            [[7, 12]],
        )

    def test_invalid_groups_are_not_decoded(self):
        response = (
            "<|mt_start|><|mt_0999|><|mt_0268|><|mt_end|>"
            "<|mt_start|><|mt_0008|><|mt_0269|><|mt_end|>"
        )
        self.assertEqual(
            parse_mask_groups(response, codebook_size=256, protocol="legacy_union"),
            [[8, 13]],
        )
        self.assertEqual(
            parse_mask_groups(response, codebook_size=256, protocol="first_mask"),
            [],
        )

    def test_protocol_validation_and_group_count(self):
        self.assertEqual(complete_mask_group_count(self.response), 2)
        self.assertEqual(validate_mask_protocol("legacy_union"), "legacy_union")
        with self.assertRaises(ValueError):
            validate_mask_protocol("unknown")

    def test_cyclegrpo_legacy_accepts_raw_pairs_and_invalid_second_code(self):
        response = "No target. <|mt_0007|><|mt_0268|><|mt_0008|><|mt_0999|>"
        self.assertEqual(cyclegrpo_legacy_raw_mask_token_count(response), 4)
        self.assertEqual(
            parse_cyclegrpo_legacy_mask_groups(response, codebook_size=256),
            [[7, 12], [8, -1]],
        )
        self.assertEqual(
            parse_mask_groups(response, codebook_size=256, protocol="legacy_union"),
            [],
        )

    def test_cyclegrpo_legacy_repairs_a_single_wrapped_token(self):
        response = "<|mt_start|><|mt_0007|><|mt_end|>"
        self.assertEqual(
            parse_cyclegrpo_legacy_mask_groups(response, codebook_size=256),
            [[7, -1]],
        )


if __name__ == "__main__":
    unittest.main()
