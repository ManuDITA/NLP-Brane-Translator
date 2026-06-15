#!/usr/bin/env python3
"""
Example usage scripts for the Healthcare Package v2.0
Demonstrates all available functions and features
"""

import json
from submodules.packages.healthcare.healthcare import HealthcareAnalyzer


def example_single_patient_analysis():
    """Demonstrate analyzing a single patient"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Patient Heart Disease Analysis")
    print("="*60)
    
    analyzer = HealthcareAnalyzer()
    
    patient_data = {
        "patient_id": "PAT001",
        "age": 65,
        "gender": "M",
        "vital_signs": {
            "blood_pressure": 160,
            "heart_rate": 85,
            "bmi": 29.5
        },
        "lab_results": {
            "total_cholesterol": 245,
            "ldl": 160,
            "hdl": 35,
            "triglycerides": 200
        },
        "medical_history": ["hypertension", "smoking", "diabetes"]
    }
    
    result = analyzer.analyze_heart_disease(patient_data)
    print(f"Patient ID: {result.patient_id}")
    print(f"Disease: {result.disease}")
    print(f"Risk Score: {result.risk_score:.2f}/100")
    print(f"Risk Level: {result.risk_level}")
    print(f"\nIdentified Risk Factors:")
    for factor in result.risk_factors:
        print(f"  • {factor}")
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  • {rec}")


def example_diabetes_analysis():
    """Demonstrate diabetes risk analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Diabetes Risk Assessment")
    print("="*60)
    
    analyzer = HealthcareAnalyzer()
    
    patient_data = {
        "patient_id": "PAT002",
        "age": 48,
        "gender": "F",
        "vital_signs": {
            "blood_pressure": 135,
            "heart_rate": 72,
            "bmi": 32.0
        },
        "lab_results": {
            "fasting_glucose": 125,
            "hba1c": 5.9,
            "total_cholesterol": 210
        },
        "medical_history": ["diabetes_family", "sedentary", "hypertension"]
    }
    
    result = analyzer.analyze_diabetes_risk(patient_data)
    print(f"Patient ID: {result.patient_id}")
    print(f"Risk Score: {result.risk_score:.2f}/100")
    print(f"Risk Level: {result.risk_level}")
    print(f"\nRisk Factors:")
    for factor in result.risk_factors:
        print(f"  • {factor}")


def example_comprehensive_report():
    """Generate comprehensive health report"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Comprehensive Health Report (All Diseases)")
    print("="*60)
    
    analyzer = HealthcareAnalyzer()
    
    patient_data = {
        "patient_id": "PAT003",
        "age": 72,
        "gender": "M",
        "vital_signs": {
            "blood_pressure": 155,
            "heart_rate": 78,
            "bmi": 27.0
        },
        "lab_results": {
            "total_cholesterol": 230,
            "fasting_glucose": 115,
            "hba1c": 5.8
        },
        "medical_history": ["hypertension", "afib", "smoking", "diabetes_family"]
    }
    
    report = analyzer.generate_health_report(patient_data)
    
    if report.get('status') == 'success':
        print(f"Patient ID: {report['patient_id']}")
        print(f"Report Timestamp: {report['timestamp']}")
        print(f"Overall Risk Level: {report['overall_risk_level'].upper()}")
        
        print("\n--- HEART DISEASE ANALYSIS ---")
        hd = report['analyses']['heart_disease']
        print(f"  Risk Score: {hd['risk_score']}")
        print(f"  Risk Level: {hd['risk_level']}")
        
        print("\n--- DIABETES ANALYSIS ---")
        dm = report['analyses']['diabetes']
        print(f"  Risk Score: {dm['risk_score']}")
        print(f"  Risk Level: {dm['risk_level']}")
        
        print("\n--- STROKE ANALYSIS ---")
        stroke = report['analyses']['stroke']
        print(f"  Risk Score: {stroke['risk_score']}")
        print(f"  Risk Level: {stroke['risk_level']}")
        print(f"  Key Risk Factors: {', '.join(stroke['risk_factors'][:3])}")


def example_data_validation():
    """Demonstrate data validation"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Data Validation")
    print("="*60)
    
    analyzer = HealthcareAnalyzer()
    
    # Valid patient
    print("Testing VALID patient data:")
    valid_patient = {
        "patient_id": "PAT004",
        "age": 45,
        "gender": "F",
        "vital_signs": {"blood_pressure": 120, "heart_rate": 70},
        "lab_results": {"total_cholesterol": 180}
    }
    is_valid, error = analyzer.validate_patient_data(valid_patient)
    print(f"  Valid: {is_valid}, Error: {error if error else 'None'}")
    
    # Invalid patient - missing field
    print("\nTesting INVALID patient (missing vital_signs):")
    invalid_patient_1 = {
        "patient_id": "PAT005",
        "age": 45,
        "gender": "F",
        "lab_results": {"total_cholesterol": 180}
    }
    is_valid, error = analyzer.validate_patient_data(invalid_patient_1)
    print(f"  Valid: {is_valid}, Error: {error}")
    
    # Invalid patient - bad age
    print("\nTesting INVALID patient (age out of range):")
    invalid_patient_2 = {
        "patient_id": "PAT006",
        "age": 200,
        "gender": "M",
        "vital_signs": {"blood_pressure": 120},
        "lab_results": {}
    }
    is_valid, error = analyzer.validate_patient_data(invalid_patient_2)
    print(f"  Valid: {is_valid}, Error: {error}")


def example_batch_processing():
    """Demonstrate batch dataset preprocessing"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Dataset Preprocessing")
    print("="*60)
    
    analyzer = HealthcareAnalyzer()
    
    # Sample dataset with mix of valid and invalid records
    dataset = [
        {
            "patient_id": "PAT007",
            "age": 55,
            "gender": "M",
            "vital_signs": {"blood_pressure": 140},
            "lab_results": {}
        },
        {
            "patient_id": "PAT008",
            "age": 45,
            "gender": "F",
            "vital_signs": {"blood_pressure": 130},
            "lab_results": {}
        },
        {
            "patient_id": "PAT009",
            "age": 999,  # Invalid age
            "gender": "M",
            "vital_signs": {"blood_pressure": 120},
            "lab_results": {}
        },
        {
            "patient_id": "PAT010",
            "age": 40
            # Missing required fields
        }
    ]
    
    result = analyzer.preprocess_dataset(dataset)
    print(f"Total Records: {result['total_records']}")
    print(f"Valid Records: {result['valid_records']}")
    print(f"Invalid Records: {result['invalid_records']}")
    print(f"Validity Rate: {result['validity_percentage']:.1f}%")
    
    if result['errors']:
        print(f"\nError Summary:")
        for error in result['errors'][:2]:  # Show first 2 errors
            print(f"  • {error['record'].get('patient_id')}: {error['error']}")


def example_stroke_risk():
    """Demonstrate stroke risk analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Stroke Risk Assessment")
    print("="*60)
    
    analyzer = HealthcareAnalyzer()
    
    patient_data = {
        "patient_id": "PAT011",
        "age": 75,
        "gender": "M",
        "vital_signs": {
            "blood_pressure": 170,
            "heart_rate": 65
        },
        "lab_results": {
            "total_cholesterol": 250
        },
        "medical_history": ["hypertension", "afib", "previous_stroke", "smoking"]
    }
    
    result = analyzer.analyze_stroke_risk(patient_data)
    print(f"Patient ID: {result.patient_id}")
    print(f"Risk Score: {result.risk_score:.2f}/100")
    print(f"Risk Level: {result.risk_level}")
    print(f"\nCritical Risk Factors:")
    for factor in result.risk_factors:
        print(f"  • {factor}")


def example_json_output_format():
    """Show JSON output format for integration"""
    print("\n" + "="*60)
    print("EXAMPLE 7: JSON Output Format")
    print("="*60)
    
    analyzer = HealthcareAnalyzer()
    
    patient_data = {
        "patient_id": "PAT012",
        "age": 60,
        "gender": "M",
        "vital_signs": {
            "blood_pressure": 150,
            "heart_rate": 80,
            "bmi": 28
        },
        "lab_results": {
            "total_cholesterol": 220,
            "fasting_glucose": 105
        },
        "medical_history": ["hypertension"]
    }
    
    result = analyzer.analyze_heart_disease(patient_data)
    output_dict = analyzer._analysis_to_dict(result)
    
    print("Output as JSON:")
    print(json.dumps(output_dict, indent=2))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HEALTHCARE PACKAGE v2.0 - USAGE EXAMPLES")
    print("="*60)
    
    example_single_patient_analysis()
    example_diabetes_analysis()
    example_comprehensive_report()
    example_data_validation()
    example_batch_processing()
    example_stroke_risk()
    example_json_output_format()
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60 + "\n")
