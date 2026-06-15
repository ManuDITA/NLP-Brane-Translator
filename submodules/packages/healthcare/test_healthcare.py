#!/usr/bin/env python3
"""
Unit tests for Healthcare Package v2.0
Tests all major functions and edge cases
"""

import unittest
import json
from submodules.packages.healthcare.healthcare import HealthcareAnalyzer, RiskLevel


class TestHealthcareAnalyzer(unittest.TestCase):
    """Test suite for HealthcareAnalyzer"""
    
    def setUp(self):
        """Initialize analyzer for each test"""
        self.analyzer = HealthcareAnalyzer()
        self.sample_patient = {
            "patient_id": "TEST001",
            "age": 55,
            "gender": "M",
            "vital_signs": {
                "blood_pressure": 140,
                "heart_rate": 75,
                "bmi": 25.0
            },
            "lab_results": {
                "total_cholesterol": 200,
                "fasting_glucose": 100,
                "hba1c": 5.5
            },
            "medical_history": []
        }
    
    # ============ Data Validation Tests ============
    
    def test_validate_patient_data_valid(self):
        """Test validation of valid patient data"""
        is_valid, error = self.analyzer.validate_patient_data(self.sample_patient)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validate_patient_data_missing_field(self):
        """Test validation fails with missing required field"""
        patient = self.sample_patient.copy()
        del patient['age']
        is_valid, error = self.analyzer.validate_patient_data(patient)
        self.assertFalse(is_valid)
        self.assertIn("age", error)
    
    def test_validate_patient_data_invalid_age(self):
        """Test validation fails with invalid age"""
        patient = self.sample_patient.copy()
        patient['age'] = 200
        is_valid, error = self.analyzer.validate_patient_data(patient)
        self.assertFalse(is_valid)
        self.assertIn("age", error)
    
    def test_validate_patient_data_invalid_gender(self):
        """Test validation fails with invalid gender"""
        patient = self.sample_patient.copy()
        patient['gender'] = 'X'
        is_valid, error = self.analyzer.validate_patient_data(patient)
        self.assertFalse(is_valid)
        self.assertIn("gender", error)
    
    # ============ Heart Disease Analysis Tests ============
    
    def test_heart_disease_low_risk(self):
        """Test heart disease analysis for low-risk patient"""
        patient = self.sample_patient.copy()
        patient['age'] = 30
        patient['medical_history'] = []
        patient['vital_signs']['blood_pressure'] = 120
        patient['lab_results']['total_cholesterol'] = 180
        
        result = self.analyzer.analyze_heart_disease(patient)
        self.assertEqual(result.disease, 'heart_disease')
        self.assertEqual(result.risk_level, RiskLevel.LOW.value)
        self.assertGreater(len(result.recommendations), 0)
    
    def test_heart_disease_high_risk(self):
        """Test heart disease analysis for high-risk patient"""
        patient = self.sample_patient.copy()
        patient['age'] = 70
        patient['gender'] = 'M'
        patient['medical_history'] = ['hypertension', 'smoking', 'diabetes']
        patient['vital_signs']['blood_pressure'] = 160
        patient['lab_results']['total_cholesterol'] = 250
        
        result = self.analyzer.analyze_heart_disease(patient)
        self.assertEqual(result.disease, 'heart_disease')
        self.assertIn(result.risk_level, [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value])
        self.assertIn("smoking", result.risk_factors[0].lower())
    
    def test_heart_disease_score_range(self):
        """Test heart disease risk score is within valid range"""
        result = self.analyzer.analyze_heart_disease(self.sample_patient)
        self.assertGreaterEqual(result.risk_score, 0.0)
        self.assertLessEqual(result.risk_score, 100.0)
    
    # ============ Diabetes Analysis Tests ============
    
    def test_diabetes_risk_low_risk(self):
        """Test diabetes analysis for low-risk patient"""
        patient = self.sample_patient.copy()
        patient['age'] = 35
        patient['vital_signs']['bmi'] = 22.0
        patient['lab_results']['fasting_glucose'] = 90
        patient['lab_results']['hba1c'] = 5.3
        
        result = self.analyzer.analyze_diabetes_risk(patient)
        self.assertEqual(result.disease, 'diabetes')
        self.assertEqual(result.risk_level, RiskLevel.LOW.value)
        self.assertLess(result.risk_score, 40)
    
    def test_diabetes_risk_high_risk(self):
        """Test diabetes analysis for high-risk patient"""
        patient = self.sample_patient.copy()
        patient['age'] = 50
        patient['vital_signs']['bmi'] = 35.0
        patient['medical_history'] = ['diabetes_family']
        patient['lab_results']['fasting_glucose'] = 130
        patient['lab_results']['hba1c'] = 6.1
        
        result = self.analyzer.analyze_diabetes_risk(patient)
        self.assertIn(result.risk_level, [RiskLevel.MODERATE.value, RiskLevel.HIGH.value])
        self.assertGreater(result.risk_score, 40)
    
    # ============ Stroke Analysis Tests ============
    
    def test_stroke_risk_low_risk(self):
        """Test stroke analysis for low-risk patient"""
        patient = self.sample_patient.copy()
        patient['age'] = 40
        patient['vital_signs']['blood_pressure'] = 120
        patient['lab_results']['total_cholesterol'] = 180
        
        result = self.analyzer.analyze_stroke_risk(patient)
        self.assertEqual(result.disease, 'stroke')
        self.assertEqual(result.risk_level, RiskLevel.LOW.value)
    
    def test_stroke_risk_high_risk(self):
        """Test stroke analysis for high-risk patient"""
        patient = self.sample_patient.copy()
        patient['age'] = 75
        patient['medical_history'] = ['hypertension', 'afib', 'previous_stroke', 'smoking']
        patient['vital_signs']['blood_pressure'] = 170
        
        result = self.analyzer.analyze_stroke_risk(patient)
        self.assertIn(result.risk_level, [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value])
        self.assertGreater(result.risk_score, 60)
    
    # ============ Report Generation Tests ============
    
    def test_generate_report_success(self):
        """Test comprehensive report generation"""
        report = self.analyzer.generate_health_report(self.sample_patient)
        
        self.assertEqual(report['status'], 'success')
        self.assertEqual(report['patient_id'], 'TEST001')
        self.assertIn('timestamp', report)
        self.assertIn('analyses', report)
        self.assertIn('overall_risk_level', report)
        self.assertIn('heart_disease', report['analyses'])
        self.assertIn('diabetes', report['analyses'])
        self.assertIn('stroke', report['analyses'])
    
    def test_generate_report_invalid_data(self):
        """Test report generation with invalid data"""
        invalid_patient = {'patient_id': 'TEST002', 'age': 200}
        report = self.analyzer.generate_health_report(invalid_patient)
        
        self.assertEqual(report['status'], 'validation_failed')
        self.assertIn('error', report)
    
    # ============ Risk Level Classification Tests ============
    
    def test_risk_level_assignment(self):
        """Test risk level correctly assigned based on score"""
        self.assertEqual(
            self.analyzer._get_risk_level('heart_disease', 15),
            RiskLevel.LOW.value
        )
        self.assertEqual(
            self.analyzer._get_risk_level('heart_disease', 35),
            RiskLevel.MODERATE.value
        )
        self.assertEqual(
            self.analyzer._get_risk_level('heart_disease', 65),
            RiskLevel.HIGH.value
        )
    
    # ============ Recommendations Tests ============
    
    def test_recommendations_generated(self):
        """Test that recommendations are generated for all risk levels"""
        # Low risk
        recs = self.analyzer._get_recommendations(
            'heart_disease', RiskLevel.LOW.value, []
        )
        self.assertGreater(len(recs), 0)
        
        # High risk
        recs = self.analyzer._get_recommendations(
            'heart_disease', RiskLevel.HIGH.value, ['smoking', 'hypertension']
        )
        self.assertGreater(len(recs), 2)
        self.assertTrue(any('specialist' in r.lower() for r in recs))
    
    # ============ Dataset Preprocessing Tests ============
    
    def test_preprocess_dataset_valid_records(self):
        """Test preprocessing dataset with all valid records"""
        dataset = [
            self.sample_patient.copy(),
            self.sample_patient.copy()
        ]
        dataset[1]['patient_id'] = 'TEST002'
        
        result = self.analyzer.preprocess_dataset(dataset)
        
        self.assertEqual(result['total_records'], 2)
        self.assertEqual(result['valid_records'], 2)
        self.assertEqual(result['invalid_records'], 0)
        self.assertEqual(result['validity_percentage'], 100.0)
    
    def test_preprocess_dataset_mixed_records(self):
        """Test preprocessing dataset with mixed valid/invalid records"""
        valid_patient = self.sample_patient.copy()
        invalid_patient = {'patient_id': 'TEST003', 'age': 250}
        
        dataset = [valid_patient, invalid_patient]
        result = self.analyzer.preprocess_dataset(dataset)
        
        self.assertEqual(result['total_records'], 2)
        self.assertEqual(result['valid_records'], 1)
        self.assertEqual(result['invalid_records'], 1)
        self.assertEqual(result['validity_percentage'], 50.0)
    
    def test_preprocess_empty_dataset(self):
        """Test preprocessing empty dataset"""
        result = self.analyzer.preprocess_dataset([])
        
        self.assertEqual(result['total_records'], 0)
        self.assertEqual(result['valid_records'], 0)
        self.assertEqual(result['invalid_records'], 0)
    
    # ============ Risk Factor Recognition Tests ============
    
    def test_identifies_age_risk_factor(self):
        """Test identification of age as risk factor"""
        patient = self.sample_patient.copy()
        patient['age'] = 70
        
        result = self.analyzer.analyze_heart_disease(patient)
        self.assertTrue(any('age' in factor.lower() for factor in result.risk_factors))
    
    def test_identifies_cholesterol_risk_factor(self):
        """Test identification of cholesterol as risk factor"""
        patient = self.sample_patient.copy()
        patient['lab_results']['total_cholesterol'] = 250
        
        result = self.analyzer.analyze_heart_disease(patient)
        self.assertTrue(any('cholesterol' in factor.lower() for factor in result.risk_factors))
    
    def test_identifies_smoking_risk_factor(self):
        """Test identification of smoking as risk factor"""
        patient = self.sample_patient.copy()
        patient['medical_history'] = ['smoking']
        
        result = self.analyzer.analyze_heart_disease(patient)
        self.assertTrue(any('smoking' in factor.lower() for factor in result.risk_factors))
    
    # ============ Overall Risk Calculation Tests ============
    
    def test_overall_risk_calculation(self):
        """Test overall risk level calculation"""
        low_scores = [15, 20, 18]
        self.assertEqual(
            self.analyzer._calculate_overall_risk(low_scores),
            RiskLevel.LOW.value
        )
        
        high_scores = [70, 65, 75]
        self.assertEqual(
            self.analyzer._calculate_overall_risk(high_scores),
            RiskLevel.HIGH.value
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        self.analyzer = HealthcareAnalyzer()
    
    def test_full_patient_workflow(self):
        """Test complete patient assessment workflow"""
        patient_data = {
            "patient_id": "INTEGRATION001",
            "age": 65,
            "gender": "M",
            "vital_signs": {
                "blood_pressure": 155,
                "heart_rate": 80,
                "bmi": 28.0
            },
            "lab_results": {
                "total_cholesterol": 240,
                "fasting_glucose": 110,
                "hba1c": 5.7
            },
            "medical_history": ["hypertension", "smoking", "diabetes_family"]
        }
        
        # Step 1: Validate data
        is_valid, _ = self.analyzer.validate_patient_data(patient_data)
        self.assertTrue(is_valid)
        
        # Step 2: Generate comprehensive report
        report = self.analyzer.generate_health_report(patient_data)
        self.assertEqual(report['status'], 'success')
        
        # Step 3: Verify all analyses present
        self.assertIn('heart_disease', report['analyses'])
        self.assertIn('diabetes', report['analyses'])
        self.assertIn('stroke', report['analyses'])
        
        # Step 4: Verify risk levels assigned
        for disease in ['heart_disease', 'diabetes', 'stroke']:
            analysis = report['analyses'][disease]
            self.assertIn('risk_level', analysis)
            self.assertIn('risk_score', analysis)
            self.assertGreater(len(analysis['recommendations']), 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
