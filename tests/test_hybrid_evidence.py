"""Comparison safety and source provenance; these are not model quality tests."""
from dataclasses import replace
from decimal import Decimal
import hashlib
import unittest

from prism_model.hybrid_evidence import (EvidenceSpan, Extraction, ReferenceUse, UnresolvedValue,
                                         compare_attribute, normalize)


class HybridEvidenceTests(unittest.TestCase):
    def check(self, description, spans=(), *, reference=None, use=None, attribute='voltage_v', uncertainties=()):
        extraction = Extraction(hashlib.sha256(description.encode()).hexdigest(), 'test-model', tuple(spans),
                                use or ReferenceUse('not_used', None, 'not_permitted', .99), tuple(uncertainties))
        return compare_attribute(description, attribute, extraction, reference or [], required_kind='motor', minimum_confidence=.9)

    def span(self, text, surface, *, role='current', owner='requested_item', confidence=.99, attribute='voltage_v', start=None):
        # Explicit positions are supplied by this fixture writer, never by the
        # comparison engine. Independent model tests must generate predictions.
        start = text.index(surface) if start is None else start
        return EvidenceSpan(attribute, start, start + len(surface), surface, role, owner, confidence)

    def test_exact_units_written_numbers_and_equivalent_decimal_values(self):
        for attr, surface, expected in [('voltage_v', '230.0 V', Decimal(230)), ('power_kw', 'zero point seven five kW', Decimal('.75')),
                                         ('phases', 'three-phase', Decimal(3)), ('phases', '3~', Decimal(3)),
                                         ('length_mm', 'two hundred mm', Decimal(200)), ('gear_ratio', '20:1', Decimal(20)),
                                         ('output_type', 'pnp', 'PNP'), ('belt_profile', 'spz', 'SPZ')]:
            self.assertEqual(normalize(attr, surface), expected)

    def test_no_raw_request_search_and_type_identifier_does_not_supply_voltage(self):
        text = 'Motor type 230; current voltage 400 V.'
        identifier = self.span(text, '230', role='type', attribute=None)
        result = self.check(text, [identifier])
        self.assertEqual(result['choice'], 'missing')
        self.assertEqual(result['evidence_spans'][0]['surface'], '230')
        self.assertEqual(result['normalized_current_values'], [])

    def test_conflict_trace_points_to_original_selected_span_and_reference_field(self):
        text = 'Request 400V; a different motor uses 230V.'
        spans = [self.span(text, '400V'), self.span(text, '230V', role='background', owner='other_item')]
        use = ReferenceUse('selected', 'A1', 'permitted', .99)
        result = self.check(text, spans, reference=[{'component_id': 'A1', 'kind': 'motor', 'voltage_v': 230}], use=use)
        self.assertEqual(result['choice'], 'conflicting')
        self.assertEqual(result['trusted_reference_field'], {'component_id': 'A1', 'field': 'voltage_v', 'value': '230'})
        self.assertEqual(result['normalized_current_values'], [{'start': 8, 'end': 12, 'value': '400'}])

    def test_irrelevant_obsolete_negated_and_type_values_are_not_current_ratings(self):
        text = '400V 230V 110V 480'
        spans = [self.span(text, '400V'), self.span(text, '230V', role='obsolete'),
                 self.span(text, '110V', role='negated'), self.span(text, '480', role='type', attribute=None)]
        self.assertEqual(self.check(text, spans)['choice'], 'satisfied')

    def test_conflicting_affirmative_values_override_ambiguity(self):
        text = '400V 230V 110V'
        spans = [self.span(text, '400V'), self.span(text, '230V'), self.span(text, '110V', role='ambiguous')]
        self.assertEqual(self.check(text, spans)['choice'], 'conflicting')

    def test_ambiguous_low_confidence_or_unresolved_ownership_require_clarification(self):
        text = '400V'
        for span in [self.span(text, text, role='ambiguous'), self.span(text, text, confidence=.89),
                     self.span(text, text, owner='unresolved'), self.span(text, text, owner='other_item')]:
            self.assertEqual(self.check(text, [span])['choice'], 'unclear')
        self.assertEqual(self.check(text, [self.span(text, text)], uncertainties=['uncertain_coverage'])['choice'], 'unclear')

    def test_uncertain_ignored_fact_cannot_hide_a_possible_conflict(self):
        text = '400V 230V'
        spans = [self.span(text, '400V'), self.span(text, '230V', role='background', owner='other_item', confidence=.5)]
        self.assertEqual(self.check(text, spans)['choice'], 'unclear')

    def test_reference_fields_require_selected_component_and_permission_to_fill(self):
        components = [{'component_id': 'A1', 'kind': 'motor', 'voltage_v': 230},
                      {'component_id': 'B2', 'kind': 'motor', 'voltage_v': 400}]
        for use, expected in [(ReferenceUse('selected', 'B2', 'permitted', .99), 'satisfied'),
                              (ReferenceUse('selected', 'B2', 'not_permitted', .99), 'missing'),
                              (ReferenceUse('selected', 'B2', 'unresolved', .99), 'unclear'),
                              (ReferenceUse('unresolved', None, 'unresolved', .99), 'unclear'),
                              (ReferenceUse('selected', 'B2', 'permitted', .8), 'unclear')]:
            result = self.check('Same motor.', reference=components, use=use)
            self.assertEqual(result['choice'], expected)
            if expected == 'satisfied':
                self.assertEqual(result['trusted_reference_field']['component_id'], 'B2')

    def test_reference_background_is_not_used_and_absent_source_field_does_not_conflict(self):
        text = '400V'
        span = self.span(text, text)
        refs = [{'component_id': 'A1', 'kind': 'motor', 'voltage_v': 230}]
        self.assertEqual(self.check(text, [span], reference=refs)['choice'], 'satisfied')
        refs[0].pop('voltage_v')
        self.assertEqual(self.check(text, [span], reference=refs, use=ReferenceUse('selected', 'A1', 'permitted', .99))['choice'], 'satisfied')

    def test_unsupported_units_and_ambiguous_numbers_do_not_get_guessed(self):
        for attr, text, expected in [('power_kw', '5hp', 'missing'), ('voltage_v', '230', 'unclear'),
                                     ('voltage_v', '1,0V', 'unclear'), ('length_mm', '1,5 mm', 'unclear'),
                                     ('phases', '1.5ph', 'unclear'), ('bore_mm', '12', 'unclear'),
                                     ('output_type', 'P', 'unclear'), ('belt_profile', '5M', 'unclear'),
                                     ('power_kw', '1.5 hp', 'missing'), ('width_mm', 'M18', 'unclear')]:
            self.assertEqual(self.check(text, [self.span(text, text, attribute=attr)], attribute=attr)['choice'], expected)

    def test_overlapping_attribute_claims_and_unoffered_references_fail_explicitly(self):
        text = '400mm'
        spans = [self.span(text, text, attribute='length_mm'), self.span(text, text, attribute='diameter_mm')]
        self.assertEqual(self.check(text, spans, attribute='length_mm')['choice'], 'unclear')
        with self.assertRaisesRegex(ValueError, 'unoffered'):
            self.check(text, use=ReferenceUse('selected', 'not-offered', 'permitted', .99))

    def test_invalid_offsets_or_description_identity_are_not_accepted(self):
        text = '400V'
        good = self.span(text, text)
        with self.assertRaisesRegex(ValueError, 'source span'):
            self.check(text, [replace(good, end=3)])
        extraction = Extraction('0' * 64, 'model', (good,), ReferenceUse('not_used', None, 'not_permitted', .99))
        with self.assertRaisesRegex(ValueError, 'bound'):
            compare_attribute(text, 'voltage_v', extraction, [], required_kind='motor', minimum_confidence=.9)


if __name__ == '__main__':
    unittest.main()


class NormalizationFormsTest(unittest.TestCase):
    """Contract wording inside one selected span; closed qualifier lists only."""

    def test_qualified_spans_and_thousands_separators(self):
        for attr, surface, expected in [
                ('gear_ratio', '15:1 reduction', Decimal(15)), ('gear_ratio', '8:1 ratio', Decimal(8)),
                ('gear_ratio', '5 to one', Decimal(5)), ('gear_ratio', 'forty to one', Decimal(40)),
                ('gear_ratio', 'i=12:1', Decimal(12)), ('supply_voltage_v', '24 V DC', Decimal(24)),
                ('supply_voltage_v', '24VDC', Decimal(24)), ('width_mm', '400 mm wide', Decimal(400)),
                ('width_mm', '17 mm width', Decimal(17)), ('length_mm', '1,200 mm', Decimal(1200)),
                ('length_mm', 'L=1,000 mm', Decimal(1000)), ('length_mm', 'one thousand mm', Decimal(1000)),
                ('voltage_v', '1,000V', Decimal(1000)), ('diameter_mm', 'Ø76 mm', Decimal(76)),
                ('outer_diameter_mm', 'OD 60 mm', Decimal(60)), ('phases', 'single phase', Decimal(1)),
                ('phases', '3~,', Decimal(3)), ('belt_profile', 'HTD 5M', 'HTD5M'),
                ('belt_profile', 'HTD5M', 'HTD5M'), ('belt_profile', 'T2.5', 'T2.5'),
                ('belt_profile', 'flat', 'FLAT'), ('output_type', 'PNP output', 'PNP'),
                ('power_kw', 'three quarters of a kilowatt', Decimal('.75')),
                ('sensing_distance_mm', 'one and a half millimetres', Decimal('1.5'))]:
            self.assertEqual(normalize(attr, surface), expected, surface)

    def test_trusted_profile_with_space_matches_request_without_space(self):
        from prism_model.hybrid_evidence import normalize_source
        self.assertEqual(normalize_source('belt_profile', 'HTD 5M'), normalize('belt_profile', 'HTD5M'))

    def test_every_broad_generator_surface_normalizes_to_its_value(self):
        import random
        from prism_model import broad_data
        values = dict(power_kw=0.75, voltage_v=400, phases=3, diameter_mm=50, length_mm=1000,
                      gear_ratio=15, shaft_diameter_mm=30, width_mm=20, belt_profile='SPA', bore_mm=25,
                      outer_diameter_mm=52, sensing_distance_mm=4, supply_voltage_v=24, output_type='PNP')
        for key, value in values.items():
            expected = normalize(key, {'power_kw': '0.75 kW', 'voltage_v': '400 V', 'phases': '3-phase',
                                       'gear_ratio': '15:1', 'supply_voltage_v': '24 V'}.get(key, f'{value} mm')
                                 if key not in ('belt_profile', 'output_type') else value)
            for seed in range(200):
                # Whole labelled phrases, as a model span may include the label.
                for role in ('train', 'development'):
                    surface = broad_data.fact(key, value, random.Random(seed), role)
                    self.assertEqual(normalize(key, surface), expected, surface)


class UnaccountedQuantityTest(unittest.TestCase):
    def test_uncovered_relevant_quantity_is_reported_and_covered_one_is_not(self):
        from prism_model.hybrid_evidence import unaccounted_quantities
        text = 'Roller: diameter 89 mm. The drawing says 900 mm and the sheet says 920 mm. Plant 1100.'
        spans = [EvidenceSpan('diameter_mm', text.index('89 mm'), text.index('89 mm') + 5, '89 mm', 'current', 'requested_item', 1.),
                 EvidenceSpan('length_mm', text.index('900 mm'), text.index('900 mm') + 6, '900 mm', 'current', 'requested_item', 1.)]
        found = unaccounted_quantities(text, ['diameter_mm', 'length_mm'], spans)
        self.assertEqual([f['surface'] for f in found], ['920 mm'])
        self.assertEqual(unaccounted_quantities(text, ['power_kw'], spans), [])

    def test_uncovered_belt_profile_and_output_codes_are_reported(self):
        from prism_model.hybrid_evidence import unaccounted_quantities
        text = 'Exact replacement of 2888-D: an HTD 3M belt. Details: an T10 belt; SPA tooth also live.'
        spans = [EvidenceSpan('belt_profile', text.index('SPA'), text.index('SPA') + 9, 'SPA tooth', 'current', 'requested_item', 1.)]
        found = unaccounted_quantities(text, ['belt_profile', 'width_mm'], spans)
        self.assertEqual([(f['surface'], f['unit_class']) for f in found], [('HTD 3M', 'profile'), ('T10', 'profile')])
        # Case-sensitive codes: ordinary words such as "at 50" or "t5" in prose do not count.
        self.assertEqual(unaccounted_quantities('Rated at 50 V, part 2888-D.', ['belt_profile'], []), [])
        self.assertEqual([f['surface'] for f in unaccounted_quantities('Sensor, PNP. Old one was NPN.', ['output_type'], [])],
                         ['PNP', 'NPN'])
        self.assertEqual(unaccounted_quantities(text, ['output_type'], spans), [])
