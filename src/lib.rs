//! # CNC - Classifier Nominal Concept
//!
//! A Rust implementation of the CNC (Classifier Nominal Concept) and CNC-BPC algorithms
//! for classification of nominal (categorical) data using Formal Concept Analysis.
//!
//! ## Features
//!
//! - **CNC Algorithm**: Finds the most pertinent attribute and computes concepts for classification
//! - **CNC-BPC**: Variant focused on minority classes for imbalanced datasets
//! - **Classification Rules Module**: Extract, filter, sort, and analyze human-readable rules (see `rules` module)
//! - **ARFF Support**: Load datasets from ARFF (Attribute-Relation File Format) files
//! - **Metrics**: Comprehensive classification metrics (accuracy, precision, recall, F1, MCC, Kappa, etc.)
//!
//! ## Quick Start
//!
//! ```
//! use cnc::{from_arff_auto, cnc, display_cnc_results_consistently};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load dataset from ARFF file
//! let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
//!
//! // Run CNC algorithm
//! let result = cnc(&dataset);
//!
//! // Display results
//! display_cnc_results_consistently(&dataset, &result.concepts);
//! # Ok(())
//! # }
//! ```

pub mod core;
pub mod metrics;
pub mod rules;

// Re-export main types and functions for cleaner API
pub use core::*;
pub use rules::*;

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use super::metrics::{evaluate_cnc, display_metrics_table, display_comparison_table, ComparisonResult};
    use std::collections::HashMap;

    // Helper function to create test data
    fn create_hashmap(entries: Vec<(String, String)>) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for (key, value) in entries {
            map.insert(key, value);
        }
        map
    }

    // Helper function to run CNC and display results with metrics
    fn run_cnc_test(dataset: &NominalDataset) -> CncResult {
        dataset.display_summary();

        println!("\n--- Running CNC ---");
        let result = cnc(dataset);
        display_cnc_chosen_attribute(dataset, &result);
        display_cnc_results_consistently(dataset, &result.concepts);

        // Extract and display classification rules
        let rules = extract_rules(dataset, &result);
        display_rules(&rules);

        // Display rule statistics
        let stats = get_rules_statistics(&rules);
        println!("\n{}", stats);

        // Calculate and display classification metrics
        let metrics = evaluate_cnc(dataset, &result);
        display_metrics_table(&metrics);

        result
    }

    // Helper function to run CNC-BPC and display results with metrics
    fn run_cnc_bpc_test(dataset: &NominalDataset, n: usize) -> CncBpcResult {
        dataset.display_summary();

        println!("\n--- Running CNC-BPC (n={}) ---", n);
        let result = cnc_bpc(dataset, n);
        println!("Minority classes kept: {:?}", result.minority_classes);
        println!("Filtered: {}/{} objects ({:.1}%)",
                 result.filtered_size, result.original_size,
                 (result.filtered_size as f64 / result.original_size as f64) * 100.0);
        display_cnc_results_consistently(dataset, &result.cnc_result.concepts);

        // Calculate and display classification metrics
        let metrics = evaluate_cnc(dataset, &result.cnc_result);
        display_metrics_table(&metrics);

        result
    }

    // Helper for ARFF-based tests
    fn run_arff_cnc_test(path: &str, class_attr: Option<&str>) -> CncResult {
        let dataset = match class_attr {
            Some(attr) => from_arff(path, attr).expect("Failed to load ARFF file"),
            None => from_arff_auto(path).expect("Failed to load ARFF file"),
        };
        run_cnc_test(&dataset)
    }

    fn run_arff_cnc_bpc_test(path: &str, class_attr: Option<&str>, n: usize) -> CncBpcResult {
        let dataset = match class_attr {
            Some(attr) => from_arff(path, attr).expect("Failed to load ARFF file"),
            None => from_arff_auto(path).expect("Failed to load ARFF file"),
        };
        run_cnc_bpc_test(&dataset, n)
    }
    
    fn create_foo_dataset() -> NominalDataset {
        let objects = vec![
            "humpty".to_string(), "dumpty".to_string(), "sat".to_string(),
            "on".to_string(), "a".to_string(), "wall".to_string()
        ];
        
        let attributes = vec![
            "StringAtt1".to_string(),
            "NominalAtt1".to_string(),
            "class".to_string()
        ];
        
        let class_attribute = "class".to_string();
        
        let data = vec![
            create_hashmap(vec![
                ("StringAtt1".to_string(), "humpty".to_string()),
                ("NominalAtt1".to_string(), "g".to_string()),
                ("class".to_string(), "A".to_string())
            ]),
            create_hashmap(vec![
                ("StringAtt1".to_string(), "dumpty".to_string()),
                ("NominalAtt1".to_string(), "g".to_string()),
                ("class".to_string(), "A".to_string())
            ]),
            create_hashmap(vec![
                ("StringAtt1".to_string(), "sat".to_string()),
                ("NominalAtt1".to_string(), "r".to_string()),
                ("class".to_string(), "B".to_string())
            ]),
            create_hashmap(vec![
                ("StringAtt1".to_string(), "on".to_string()),
                ("NominalAtt1".to_string(), "r".to_string()),
                ("class".to_string(), "B".to_string())
            ]),
            create_hashmap(vec![
                ("StringAtt1".to_string(), "a".to_string()),
                ("NominalAtt1".to_string(), "g".to_string()),
                ("class".to_string(), "A".to_string())
            ]),
            create_hashmap(vec![
                ("StringAtt1".to_string(), "wall".to_string()),
                ("NominalAtt1".to_string(), "r".to_string()),
                ("class".to_string(), "B".to_string())
            ]),
        ];
        
        NominalDataset::new(objects, attributes, class_attribute, data)
    }
    
    fn create_animal_dataset() -> NominalDataset {
        let animal_objects = vec![
            "animal1".to_string(), "animal2".to_string(), "animal3".to_string(),
            "animal4".to_string(), "animal5".to_string()
        ];
        
        let animal_attributes = vec![
            "has_fur".to_string(),
            "can_fly".to_string(), 
            "lives_in_water".to_string(),
            "class".to_string()
        ];
        
        let animal_data = vec![
            create_hashmap(vec![
                ("has_fur".to_string(), "yes".to_string()),
                ("can_fly".to_string(), "no".to_string()),
                ("lives_in_water".to_string(), "no".to_string()),
                ("class".to_string(), "mammal".to_string())
            ]),
            create_hashmap(vec![
                ("has_fur".to_string(), "no".to_string()),
                ("can_fly".to_string(), "yes".to_string()),
                ("lives_in_water".to_string(), "no".to_string()),
                ("class".to_string(), "bird".to_string())
            ]),
            create_hashmap(vec![
                ("has_fur".to_string(), "no".to_string()),
                ("can_fly".to_string(), "no".to_string()),
                ("lives_in_water".to_string(), "yes".to_string()),
                ("class".to_string(), "fish".to_string())
            ]),
            create_hashmap(vec![
                ("has_fur".to_string(), "yes".to_string()),
                ("can_fly".to_string(), "no".to_string()),
                ("lives_in_water".to_string(), "yes".to_string()),
                ("class".to_string(), "mammal".to_string())
            ]),
            create_hashmap(vec![
                ("has_fur".to_string(), "no".to_string()),
                ("can_fly".to_string(), "yes".to_string()),
                ("lives_in_water".to_string(), "yes".to_string()),
                ("class".to_string(), "bird".to_string())
            ]),
        ];
        
        NominalDataset::new(animal_objects, animal_attributes, "class".to_string(), animal_data)
    }
    
    fn create_weather_dataset() -> NominalDataset {
        let objects = vec![
            "o2".to_string(), "o6".to_string(), "o9".to_string(),
            "o10".to_string(), "o13".to_string()
        ];
        
        let attributes = vec![
            "Outlook".to_string(),
            "Temperature".to_string(),
            "Humidity".to_string(),
            "Windy".to_string(),
            "Play".to_string(),
        ];
        
        let class_attribute = "Play".to_string();
        
        let data = vec![
            create_hashmap(vec![
                ("Outlook".to_string(), "Sunny".to_string()),
                ("Temperature".to_string(), "Hot".to_string()),
                ("Humidity".to_string(), "High".to_string()),
                ("Windy".to_string(), "True".to_string()),
                ("Play".to_string(), "No".to_string())
            ]),
            create_hashmap(vec![
                ("Outlook".to_string(), "Rainy".to_string()),
                ("Temperature".to_string(), "Cool".to_string()),
                ("Humidity".to_string(), "Normal".to_string()),
                ("Windy".to_string(), "True".to_string()),
                ("Play".to_string(), "No".to_string())
            ]),
            create_hashmap(vec![
                ("Outlook".to_string(), "Sunny".to_string()),
                ("Temperature".to_string(), "Cool".to_string()),
                ("Humidity".to_string(), "Normal".to_string()),
                ("Windy".to_string(), "False".to_string()),
                ("Play".to_string(), "Yes".to_string())
            ]),
            create_hashmap(vec![
                ("Outlook".to_string(), "Rainy".to_string()),
                ("Temperature".to_string(), "Mild".to_string()),
                ("Humidity".to_string(), "Normal".to_string()),
                ("Windy".to_string(), "False".to_string()),
                ("Play".to_string(), "Yes".to_string())
            ]),
            create_hashmap(vec![
                ("Outlook".to_string(), "Overcast".to_string()),
                ("Temperature".to_string(), "Hot".to_string()),
                ("Humidity".to_string(), "Normal".to_string()),
                ("Windy".to_string(), "False".to_string()),
                ("Play".to_string(), "Yes".to_string())
            ]),
        ];
        
        NominalDataset::new(objects, attributes, class_attribute, data)
    }

    #[test]
    fn cnc_foo() {
        let dataset = create_foo_dataset();
        let result = run_cnc_test(&dataset);
        assert_eq!(8, result.concepts.len());
    }

    #[test]
    fn cnc_animal() {
        let dataset = create_animal_dataset();
        let result = run_cnc_test(&dataset);
        assert_eq!(2, result.concepts.len());
    }

    #[test]
    fn cnc_weather() {
        let dataset = create_weather_dataset();
        let result = run_cnc_test(&dataset);
        assert_eq!(1, result.concepts.len());
    }

    #[test]
    fn test_classification_rules_extraction() {
        println!("\n=== Test: Classification Rules Extraction ===\n");

        let dataset = create_foo_dataset();
        dataset.display_summary();

        // Run CNC
        let result = cnc(&dataset);

        // Extract rules
        let rules = extract_rules(&dataset, &result);
        println!("\n--- Classification Rules (compact) ---");
        display_rules(&rules);

        // Detailed rules display
        println!("\n--- Classification Rules (detailed) ---");
        display_rules_detailed(&dataset, &rules);

        // Filter rules by confidence
        println!("\n--- Filtering Rules ---");
        let high_conf_rules = filter_rules_by_confidence(&rules, 100.0);
        println!("Rules with 100% confidence: {}", high_conf_rules.len());

        let medium_conf_rules = filter_rules_by_confidence(&rules, 50.0);
        println!("Rules with ≥50% confidence: {}", medium_conf_rules.len());

        // Sort by confidence
        let mut sorted_rules = rules.clone();
        sort_rules_by_confidence(&mut sorted_rules);
        println!("\n--- Top 3 rules by confidence ---");
        for (i, rule) in sorted_rules.iter().take(3).enumerate() {
            println!("{}. {}", i + 1, rule);
        }

        // Statistics
        let stats = get_rules_statistics(&rules);
        println!("\n{}", stats);

        assert_eq!(8, rules.len());
    }

    #[test]
    fn test_rule_matching() {
        println!("\n=== Test: Rule Matching ===\n");

        let dataset = create_animal_dataset();
        let result = cnc(&dataset);
        let rules = extract_rules(&dataset, &result);

        println!("Testing rule matching on dataset objects:");

        for (obj_idx, obj_name) in dataset.objects.iter().enumerate() {
            let obj_data = &dataset.data[obj_idx];
            let actual_class = obj_data.get(&dataset.class_attribute).unwrap();

            println!("\nObject '{}' (actual class: {})", obj_name, actual_class);

            for (rule_idx, rule) in rules.iter().enumerate() {
                if rule.matches(obj_data) {
                    println!("  ✓ Matched by Rule {}: predicts '{}'",
                             rule_idx + 1, rule.predicted_class);
                }
            }
        }
    }

    #[test]
    fn cnc_bpc_weather() {
        let dataset = create_weather_dataset();
        run_cnc_bpc_test(&dataset, 1);
    }

    /*
    // Test cases: (n, expected_concepts, check_type)
    // check_type: Some(n) = exact match, None = less than previous
    #[test]
    fn cnc_bp_foo() {
        let dataset = create_foo_dataset();
        let test_cases = [(3, 8), (5, 8), (2, 8)]; // (n, expected)

        for (n, expected) in test_cases {
            let result = run_cnc_bp_test(&dataset, "Foo", n);
            assert_eq!(expected, result.cnc_result.concepts.len());
        }

        // n=1 should have fewer concepts
        let result = run_cnc_bp_test(&dataset, "Foo", 1);
        assert!(result.cnc_result.concepts.len() < 8);
    }

    #[test]
    fn cnc_bp_animal() {
        let dataset = create_animal_dataset();
        let test_cases = [(3, 2), (2, 2), (1, 2)];

        for (n, expected) in test_cases {
            let result = run_cnc_bp_test(&dataset, "Animal", n);
            assert_eq!(expected, result.cnc_result.concepts.len());
        }
    }

    #[test]
    fn cnc_bp_weather() {
        let dataset = create_weather_dataset();
        let test_cases = [(2, 1), (1, 1)];

        for (n, expected) in test_cases {
            let result = run_cnc_bp_test(&dataset, "Weather", n);
            assert_eq!(expected, result.cnc_result.concepts.len());
        }
    }
    */

    // Weather.
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t01_cnc_arff_weather() {
        run_arff_cnc_test("data-examples/weather.nominal.arff", None);
    }
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t02_cnc_bpc_arff_weather() {
        run_arff_cnc_bpc_test("data-examples/weather.nominal.arff", None, 1);
    }

    // Contact Lenses
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t03_cnc_arff_contact_lenses() {
        run_arff_cnc_test("data-examples/contact-lenses.arff", Some("contact-lenses"));
    }
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t04_cnc_bpc_arff_contact_lenses_1() {
        run_arff_cnc_bpc_test("data-examples/contact-lenses.arff", Some("contact-lenses"), 1);
    }
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t05_cnc_bpc_arff_contact_lenses_2() {
        run_arff_cnc_bpc_test("data-examples/contact-lenses.arff", Some("contact-lenses"), 2);
    }

    // Breast Cancer
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t06_cnc_arff_breast_cancer() {
        run_arff_cnc_test("data-examples/breast-cancer.arff", Some("Class"));
    }
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t07_cnc_bpc_arff_breast_cancer_1() {
        run_arff_cnc_bpc_test("data-examples/breast-cancer.arff", Some("Class"), 1);
    }
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t08_cnc_bpc_arff_breast_cancer_2() {
        run_arff_cnc_bpc_test("data-examples/breast-cancer.arff", Some("Class"), 2);
    }

    // Unbalanced.
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t09_cnc_arff_unbalanced() {
        run_arff_cnc_test("data-examples/unbalanced.arff", None);
    }
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t10_cnc_bpc_arff_unbalanced_1() {
        run_arff_cnc_bpc_test("data-examples/unbalanced.arff", None, 1);
    }
    #[test]
    #[ignore] // Nécessite fichier .arff
    fn t11_cnc_bpc_arff_unbalanced_2() {
        run_arff_cnc_bpc_test("data-examples/unbalanced.arff", None, 2);
    }

    // Comparison summary test - runs last (z prefix for ordering)
    #[test]
    #[ignore] // Nécessite fichiers .arff
    fn z_comparison_summary() {
        println!("\n\n{}", "=".repeat(80));
        println!("                    COMPARISON SUMMARY: CNC vs CNC-BPC");
        println!("{}\n", "=".repeat(80));

        let datasets: Vec<(&str, Option<&str>, usize)> = vec![
            ("data-examples/weather.nominal.arff", None, 1),
            ("data-examples/contact-lenses.arff", Some("contact-lenses"), 1),
            ("data-examples/breast-cancer.arff", Some("Class"), 1),
            ("data-examples/unbalanced.arff", None, 1),
        ];

        let mut comparisons = Vec::new();

        for (path, class_attr, n) in datasets {
            let dataset = match class_attr {
                Some(attr) => from_arff(path, attr),
                None => from_arff_auto(path),
            };

            if let Ok(dataset) = dataset {
                // Extract dataset name from path
                let name = path.split('/').last().unwrap_or(path)
                    .replace(".arff", "")
                    .replace(".nominal", "");

                // Run CNC
                let cnc_result = cnc(&dataset);
                let cnc_metrics = evaluate_cnc(&dataset, &cnc_result);

                // Run CNC-BPC
                let cnc_bpc_result = cnc_bpc(&dataset, n);
                let cnc_bpc_metrics = evaluate_cnc(&dataset, &cnc_bpc_result.cnc_result);

                comparisons.push(ComparisonResult {
                    dataset_name: name,
                    cnc_metrics,
                    cnc_bpc_metrics,
                    cnc_bpc_n: n,
                });
            }
        }

        display_comparison_table(&comparisons);
    }
}
