//! CNC (Classifier Nominal Concept) algorithm and classification rule extraction.
//!
//! This module implements the CNC and CNC-BPC algorithms for classification of nominal
//! (categorical) data using Formal Concept Analysis. It also provides comprehensive
//! tools for extracting, filtering, sorting, and analyzing classification rules.
//!
//! # Overview
//!
//! The CNC algorithm finds the most pertinent attribute in a dataset and computes
//! concepts (extent/intent pairs) that can be used for classification. CNC-BPC is
//! a variant that focuses on minority classes for imbalanced datasets.
//!
//! # Classification Rules
//!
//! Rules extracted from CNC concepts have the form:
//! ```text
//! IF condition1 AND condition2 ... THEN class = X (confidence Y%, support N)
//! ```
//!
//! Each rule contains:
//! - **Conditions**: Attribute-value pairs (the concept's intent)
//! - **Predicted class**: Majority class in the concept's extent
//! - **Confidence**: Percentage of the majority class (quality indicator)
//! - **Support**: Number of objects covered (generality indicator)
//!
//! # Quick Start
//!
//! ```
//! use fcars::cnc::{from_arff_auto, cnc, extract_rules, display_rules};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // 1. Load dataset
//! let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
//!
//! // 2. Run CNC algorithm
//! let result = cnc(&dataset);
//!
//! // 3. Extract classification rules
//! let rules = extract_rules(&dataset, &result);
//!
//! // 4. Display rules
//! display_rules(&rules);
//! # Ok(())
//! # }
//! ```
//!
//! # Working with Rules
//!
//! ## Filtering Rules
//!
//! ```
//! use fcars::cnc::{extract_rules, filter_rules_by_confidence, filter_rules_by_support};
//! # use fcars::cnc::{from_arff_auto, cnc};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
//! # let result = cnc(&dataset);
//! # let rules = extract_rules(&dataset, &result);
//!
//! // Keep only high-confidence rules
//! let reliable_rules = filter_rules_by_confidence(&rules, 80.0);
//!
//! // Keep only general rules (covering many objects)
//! let general_rules = filter_rules_by_support(&rules, 5);
//! # Ok(())
//! # }
//! ```
//!
//! ## Sorting Rules
//!
//! ```
//! use fcars::cnc::{extract_rules, sort_rules_by_confidence, sort_rules_by_support};
//! # use fcars::cnc::{from_arff_auto, cnc};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
//! # let result = cnc(&dataset);
//! # let mut rules = extract_rules(&dataset, &result);
//!
//! // Sort by confidence (best rules first)
//! sort_rules_by_confidence(&mut rules);
//!
//! // Or sort by support (most general rules first)
//! sort_rules_by_support(&mut rules);
//! # Ok(())
//! # }
//! ```
//!
//! ## Classifying New Objects
//!
//! ```
//! use fcars::cnc::extract_rules;
//! use std::collections::HashMap;
//! # use fcars::cnc::{from_arff_auto, cnc};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
//! # let result = cnc(&dataset);
//! # let rules = extract_rules(&dataset, &result);
//!
//! // Create a new object to classify
//! let mut new_object = HashMap::new();
//! new_object.insert("outlook".to_string(), "sunny".to_string());
//! new_object.insert("humidity".to_string(), "high".to_string());
//!
//! // Find matching rules
//! for rule in &rules {
//!     if rule.matches(&new_object) {
//!         println!("Predicted class: {}", rule.predicted_class);
//!         println!("Confidence: {:.1}%", rule.confidence);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Analyzing Rule Sets
//!
//! ```
//! use fcars::cnc::{extract_rules, get_rules_statistics};
//! # use fcars::cnc::{from_arff_auto, cnc};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
//! # let result = cnc(&dataset);
//! # let rules = extract_rules(&dataset, &result);
//!
//! let stats = get_rules_statistics(&rules);
//! println!("{}", stats);
//! // Displays aggregate metrics: avg confidence, support, etc.
//! # Ok(())
//! # }
//! ```
//!
//! # CNC-BPC for Imbalanced Data
//!
//! CNC-BPC focuses on minority classes, useful for imbalanced datasets:
//!
//! ```
//! use fcars::cnc::{from_arff_auto, cnc_bpc, extract_rules};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
//!
//! // Run CNC-BPC keeping 1 minority class
//! let result = cnc_bpc(&dataset, 1);
//!
//! println!("Minority classes: {:?}", result.minority_classes);
//!
//! // Extract rules focused on minority class
//! let rules = extract_rules(&dataset, &result.cnc_result);
//! # Ok(())
//! # }
//! ```
//!
//! # Rule Quality Metrics
//!
//! - **Confidence**: Higher is better (indicates rule reliability)
//! - **Support**: Higher = more general, lower = more specific
//! - **Coverage**: Percentage of dataset covered by the rule
//! - **Number of conditions**: Fewer = simpler/general, more = complex/specific
//!
//! # See Also
//!
//! - [`ClassificationRule`] - The rule structure
//! - [`extract_rules`] - Extract rules from concepts
//! - [`display_rules`] - Display rules (compact format)
//! - [`display_rules_detailed`] - Display rules (detailed format)
//! - [`get_rules_statistics`] - Compute aggregate statistics

use std::collections::{HashMap, HashSet};
use std::fs;

/// Classification rule extracted from a CNC/CNC-BPC concept.
///
/// A classification rule represents an if-then pattern for classification:
/// **IF** conditions **THEN** class = X (with confidence Y%)
///
/// # Structure
///
/// Each rule contains:
/// - **Conditions**: Attribute-value pairs from the concept's intent
/// - **Predicted class**: The majority class in the concept's extent
/// - **Confidence**: Percentage of the majority class (0-100%)
/// - **Support**: Number of objects covered by this rule
/// - **Coverage**: Percentage of objects covered in the dataset
///
/// # Example
///
/// ```
/// use fcars::cnc::{NominalDataset, cnc, extract_rules};
/// use std::collections::HashMap;
///
/// // Create a simple dataset
/// let objects = vec!["obj1".to_string(), "obj2".to_string(), "obj3".to_string()];
/// let attributes = vec!["color".to_string(), "size".to_string(), "class".to_string()];
///
/// let mut data = vec![];
/// for _ in 0..3 {
///     let mut obj = HashMap::new();
///     obj.insert("color".to_string(), "red".to_string());
///     obj.insert("size".to_string(), "small".to_string());
///     obj.insert("class".to_string(), "A".to_string());
///     data.push(obj);
/// }
///
/// let dataset = NominalDataset::new(objects, attributes, "class".to_string(), data);
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// // Use the rules
/// for rule in &rules {
///     println!("Rule: {}", rule);
///     println!("  Confidence: {:.1}%", rule.confidence);
///     println!("  Support: {}", rule.support);
///     println!("  Coverage: {:.1}%", rule.coverage());
/// }
/// ```
///
/// # Quality Metrics
///
/// - **Confidence**: Proportion of the predicted class among covered objects (higher is better)
/// - **Support**: Number of objects covered (low support may indicate noise)
/// - **Coverage**: Percentage of dataset covered (general vs specific rules)
/// - **Number of conditions**: Rule complexity (fewer = more general, more = more specific)
#[derive(Debug, Clone)]
pub struct ClassificationRule {
    /// Conditions (attribute-value pairs from the intent)
    pub conditions: HashMap<String, String>,
    /// Predicted class (majority class in the extent)
    pub predicted_class: String,
    /// Confidence: percentage of the majority class in the extent
    pub confidence: f64,
    /// Support: number of objects covered by this rule
    pub support: usize,
    /// Total objects in dataset
    pub total_objects: usize,
    /// Object indices covered by this rule
    pub covered_objects: Vec<usize>,
    /// Name of the class attribute
    pub class_attribute_name: String,
}

impl ClassificationRule {
    /// Returns the coverage percentage of this rule in the dataset.
    ///
    /// Coverage represents the proportion of objects in the dataset that are
    /// covered by this rule.
    ///
    /// # Example
    ///
    /// ```
    /// # use fcars::cnc::{NominalDataset, cnc, extract_rules};
    /// # use std::collections::HashMap;
    /// # let objects = vec!["obj1".to_string()];
    /// # let attributes = vec!["attr".to_string(), "class".to_string()];
    /// # let mut data = vec![];
    /// # let mut obj = HashMap::new();
    /// # obj.insert("attr".to_string(), "val".to_string());
    /// # obj.insert("class".to_string(), "A".to_string());
    /// # data.push(obj);
    /// # let dataset = NominalDataset::new(objects, attributes, "class".to_string(), data);
    /// # let result = cnc(&dataset);
    /// # let rules = extract_rules(&dataset, &result);
    /// for rule in &rules {
    ///     if rule.coverage() > 50.0 {
    ///         println!("High coverage rule: {:.1}%", rule.coverage());
    ///     }
    /// }
    /// ```
    pub fn coverage(&self) -> f64 {
        (self.support as f64 / self.total_objects as f64) * 100.0
    }

    /// Checks if this rule matches a given object.
    ///
    /// A rule matches an object if all conditions (attribute-value pairs) of the rule
    /// are satisfied by the object's attributes.
    ///
    /// # Arguments
    ///
    /// * `object_data` - HashMap containing the object's attribute values
    ///
    /// # Returns
    ///
    /// `true` if all rule conditions match the object, `false` otherwise
    ///
    /// # Example
    ///
    /// ```
    /// # use fcars::cnc::{NominalDataset, cnc, extract_rules};
    /// # use std::collections::HashMap;
    /// # let objects = vec!["obj1".to_string()];
    /// # let attributes = vec!["color".to_string(), "class".to_string()];
    /// # let mut data = vec![];
    /// # let mut obj_data = HashMap::new();
    /// # obj_data.insert("color".to_string(), "red".to_string());
    /// # obj_data.insert("class".to_string(), "A".to_string());
    /// # data.push(obj_data.clone());
    /// # let dataset = NominalDataset::new(objects, attributes, "class".to_string(), data);
    /// # let result = cnc(&dataset);
    /// # let rules = extract_rules(&dataset, &result);
    /// // Create a new object to classify
    /// let mut new_object = HashMap::new();
    /// new_object.insert("color".to_string(), "red".to_string());
    /// new_object.insert("size".to_string(), "small".to_string());
    ///
    /// // Find matching rules
    /// for rule in &rules {
    ///     if rule.matches(&new_object) {
    ///         println!("Match! Predicted class: {}", rule.predicted_class);
    ///     }
    /// }
    /// ```
    pub fn matches(&self, object_data: &HashMap<String, String>) -> bool {
        self.conditions.iter().all(|(attr, value)| {
            object_data.get(attr).map(|v| v == value).unwrap_or(false)
        })
    }
}

impl std::fmt::Display for ClassificationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Format conditions
        if self.conditions.is_empty() {
            write!(f, "IF <no conditions>")?;
        } else {
            write!(f, "IF ")?;
            let mut conds: Vec<_> = self.conditions.iter().collect();
            conds.sort_by_key(|(k, _)| *k);

            for (i, (attr, value)) in conds.iter().enumerate() {
                if i > 0 {
                    write!(f, " AND ")?;
                }
                write!(f, "{}={}", attr, value)?;
            }
        }

        write!(f, " THEN {}={} ", self.class_attribute_name, self.predicted_class)?;
        write!(f, "(confidence={:.1}%, support={}/{}, coverage={:.1}%)",
               self.confidence, self.support, self.total_objects, self.coverage())
    }
}

/// Result structure for CNC containing both the concepts and debug information
#[derive(Debug)]
pub struct CncResult {
    pub concepts: Vec<(String, String, Vec<usize>, HashMap<String, String>)>,
    pub pertinent_attrs: Vec<String>,
}

/// Result structure for CNC-BPC containing both the concepts and debug information
/// Uses CncResult to avoid duplication and maintain consistency
#[derive(Debug)]
pub struct CncBpcResult {
    pub cnc_result: CncResult,
    pub minority_classes: HashSet<String>,
    pub original_size: usize,
    pub filtered_size: usize,
}

/// CNC (Classifier Nominal Concept)
/// A classifier that uses Formal Concept Analysis to extract concepts from nominal (multi-valued) data.
/// The algorithm finds the most pertinent attribute and computes its closure (concept).

/// Nominal dataset structure for CNC
#[derive(Debug, Clone)]
pub struct NominalDataset {
    pub objects: Vec<String>,
    pub attributes: Vec<String>,
    pub class_attribute: String,
    pub data: Vec<HashMap<String, String>>, // Each object's attribute values
}

impl std::fmt::Display for NominalDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print header - include all attributes with class clearly separated
        write!(f, "{:>10}", "")?;
        for attr in &self.attributes {
            if attr != &self.class_attribute {
                write!(f, "{:>15}", attr)?;
            }
        }
        // Add visual separator and class column together
        write!(f, "       │  {:>8}", self.class_attribute)?;
        writeln!(f)?;
        
        // Print separator
        write!(f, "{:>10}", "")?;
        for attr in &self.attributes {
            if attr != &self.class_attribute {
                write!(f, "{:>15}", "")?;
            }
        }
        // Add visual separator in the separator line
        write!(f, "       │  {:>8}", "")?;
        writeln!(f)?;
        
        // Print each row
        for (i, obj) in self.objects.iter().enumerate() {
            write!(f, "{:>10}", obj)?;
            // Print descriptive attributes (exclude class)
            for attr in &self.attributes {
                if attr != &self.class_attribute {
                    if let Some(val) = self.data[i].get(attr) {
                        write!(f, "{:>15}", val)?;
                    } else {
                        write!(f, "{:>15}", "?")?;
                    }
                }
            }
            // Add visual separator and class value together
            write!(f, "       │  ")?;
            if let Some(class_val) = self.data[i].get(&self.class_attribute) {
                write!(f, "{:>8}", class_val)?;
            } else {
                write!(f, "{:>8}", "?")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl NominalDataset {
    /// Create a new nominal dataset
    pub fn new(objects: Vec<String>, attributes: Vec<String>, class_attribute: String, data: Vec<HashMap<String, String>>) -> Self {
        Self {
            objects,
            attributes,
            class_attribute,
            data,
        }
    }
    
    /// Get all unique values for an attribute
    pub fn get_attribute_values(&self, attr_name: &str) -> Vec<String> {
        let mut values = HashSet::new();
        for obj_data in &self.data {
            if let Some(val) = obj_data.get(attr_name) {
                values.insert(val.clone());
            }
        }
        let mut result: Vec<String> = values.into_iter().collect();
        result.sort();
        result
    }
    
    /// Group objects by attribute value
    pub fn group_by_attribute_value(&self, attr_name: &str) -> HashMap<String, Vec<usize>> {
        let mut groups = HashMap::new();
        for (obj_idx, obj_data) in self.data.iter().enumerate() {
            if let Some(val) = obj_data.get(attr_name) {
                groups.entry(val.clone()).or_insert_with(Vec::new).push(obj_idx);
            }
        }
        groups
    }
    
    /// Get class values for a group of objects
    pub fn get_class_values(&self, object_indices: &[usize]) -> Vec<String> {
        object_indices.iter()
            .filter_map(|&obj_idx| self.data[obj_idx].get(&self.class_attribute).cloned())
            .collect()
    }
    
    /// Get the majority class from a class distribution
    /// Returns (majority_class, count, percentage)
    pub fn get_majority_class(class_values: &[String]) -> Option<(String, usize, f64)> {
        if class_values.is_empty() {
            return None;
        }
        
        let mut class_counts = HashMap::new();
        
        // Count occurrences of each class
        for class_val in class_values {
            *class_counts.entry(class_val.clone()).or_insert(0) += 1;
        }
        
        // Find the class with maximum count
        let (majority_class, count) = class_counts.into_iter()
            .max_by_key(|(_, count)| *count)?;
        
        let percentage = (count as f64 / class_values.len() as f64) * 100.0;
        
        Some((majority_class, count, percentage))
    }
    
    /// Display summary statistics about the dataset
    pub fn display_summary(&self) {

        println!("Context:\n{}", &self);

        // Count descriptive attributes (excluding class)
        let desc_attrs: Vec<_> = self.attributes.iter()
            .filter(|attr| attr != &&self.class_attribute)
            .collect();
            
        println!("Dataset Summary:");
        println!("- Objects: {}", self.objects.len());
        println!("- Descriptive attributes: {}", desc_attrs.len());
        println!("- Class attribute: {}", self.class_attribute);
        
        for attr in &desc_attrs {
            let values = self.get_attribute_values(attr);
            println!("- Attribute '{}': {} unique values", attr, values.len());
        }
        
        // Show class distribution
        let class_values = self.get_class_values(&(0..self.objects.len()).collect::<Vec<_>>());
        let mut class_counts = HashMap::new();
        for class_val in class_values {
            *class_counts.entry(class_val).or_insert(0) += 1;
        }
        
        println!("- Class distribution:");
        for (class_val, count) in class_counts {
            println!("  {}: {} ({:.1}%)", class_val, count, (count as f64 / self.objects.len() as f64) * 100.0);
        }
    }
}

/// Calculate entropy for nominal values
fn calculate_entropy(values: &[String]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let total = values.len() as f64;
    let mut value_counts = HashMap::new();
    
    // Count occurrences of each value
    for value in values {
        *value_counts.entry(value.clone()).or_insert(0) += 1;
    }
    
    // Calculate entropy: -Σ p_i * log2(p_i)
    value_counts.values().map(|&count| {
        let probability = count as f64 / total;
        -probability * probability.log2()
    }).sum()
}

/// Calculate information gain for a nominal attribute
fn information_gain(dataset: &NominalDataset, attr_name: &str) -> f64 {
    let total_objects = dataset.objects.len() as f64;
    if total_objects == 0.0 {
        return 0.0;
    }
    
    // Get all class values for total entropy calculation
    let all_class_values = dataset.get_class_values(&(0..dataset.objects.len()).collect::<Vec<_>>());
    let total_entropy = calculate_entropy(&all_class_values);
    
    // Group by attribute value and calculate weighted entropy
    let groups = dataset.group_by_attribute_value(attr_name);
    let mut weighted_entropy = 0.0;
    
    for (_, object_indices) in &groups {
        let group_size = object_indices.len() as f64;
        let weight = group_size / total_objects;
        
        // Get class values for this group
        let group_class_values = dataset.get_class_values(object_indices);
        let group_entropy = calculate_entropy(&group_class_values);
        weighted_entropy += weight * group_entropy;
    }
    
    total_entropy - weighted_entropy
}

/// Find the most pertinent attribute using information gain
/// Returns all attributes with maximum gain (handles ties)
fn find_most_pertinent_attributes(dataset: &NominalDataset) -> Vec<String> {
    let mut attr_gains = Vec::new();
    
    for attr_name in &dataset.attributes {
        if attr_name == &dataset.class_attribute {
            continue; // Skip class attribute
        }
        
        let gain = information_gain(dataset, attr_name);
        attr_gains.push((attr_name.clone(), gain));
    }
    
    // Find maximum gain
    let max_gain = attr_gains.iter()
        .map(|(_, gain)| *gain)
        .fold(f64::MIN, f64::max);
    
    // Return all attributes with maximum gain
    attr_gains.into_iter()
        .filter(|(_, gain)| *gain == max_gain)
        .map(|(attr, _)| attr)
        .collect()
}

/// Find the most frequent values for an attribute
/// Returns all values with maximum frequency (handles ties)
fn find_most_frequent_values(dataset: &NominalDataset, attr_name: &str) -> Vec<String> {
    let groups = dataset.group_by_attribute_value(attr_name);
    
    if groups.is_empty() {
        return Vec::new();
    }
    
    // Find maximum frequency
    let max_freq = groups.values()
        .map(|indices| indices.len())
        .max()
        .unwrap_or(0);
    
    // Return all values with maximum frequency
    groups.into_iter()
        .filter(|(_, indices)| indices.len() == max_freq)
        .map(|(value, _)| value)
        .collect()
}

/// Compute closure for nominal data (group objects by attribute value and find common attributes)
pub fn compute_nominal_closure(dataset: &NominalDataset, attr_name: &str, attr_value: &str) -> (Vec<usize>, HashMap<String, String>) {
    // Step 1: Find all objects with this attribute value (extent)
    let extent: Vec<usize> = dataset.data.iter()
        .enumerate()
        .filter(|(_, obj_data)| obj_data.get(attr_name) == Some(&attr_value.to_string()))
        .map(|(idx, _)| idx)
        .collect();
    
    // Step 2: Find common attributes for these objects (intent)
    if extent.is_empty() {
        return (extent, HashMap::new());
    }
    
    let mut intent = HashMap::new();
    let first_obj = &dataset.data[extent[0]];
    
    // Check which attributes have the same value across all objects in extent
    // Only include descriptive attributes (exclude class) in the intent
    for attr in &dataset.attributes {
        if attr == &dataset.class_attribute {
            continue; // Skip class attribute - it's not part of the formal context
        }
        
        let first_value = first_obj.get(attr);
        if first_value.is_none() {
            continue;
        }
        
        let all_same = extent.iter().all(|&obj_idx| {
            dataset.data[obj_idx].get(attr) == first_value
        });
        
        if all_same {
            intent.insert(attr.clone(), first_value.unwrap().clone());
        }
    }
    
    (extent, intent)
}

/// CNC algorithm.
/// Returns all concepts when there are ties in pertinence or frequency
pub fn cnc(dataset: &NominalDataset) -> CncResult {
    
    // Step 1: Find all most pertinent attributes (handle ties)
    let pertinent_attrs = find_most_pertinent_attributes(dataset);
    
    if pertinent_attrs.is_empty() {
        return CncResult {
            concepts: Vec::new(),
            pertinent_attrs: Vec::new(),
        };
    }
    
    // Steps 2 and 3 there.
    cnc_core(pertinent_attrs, dataset)
}

fn cnc_core(pertinent_attrs: Vec<String>, dataset: &NominalDataset) -> CncResult {
    let mut results = Vec::new();

    // Step 2 of the CNC algorithm: For each pertinent attribute, find all most frequent values (handle ties)
    for pertinent_attr in &pertinent_attrs {
        let most_frequent_values = find_most_frequent_values(dataset, pertinent_attr);
        
        // Step 3: Compute closure for each attribute-value pair
        for value in &most_frequent_values {
            let (extent, intent) = compute_nominal_closure(dataset, pertinent_attr, value);
            results.push((pertinent_attr.clone(), value.clone(), extent, intent));
        }
    }
    
    CncResult {
        concepts: results,
        pertinent_attrs,
    }
}

/// CNC-BPC: CNC Bottom-Pertinent Classes - focuses on minority classes for imbalanced datasets.
///
/// This variant of CNC filters the dataset to keep only the `n` most minority (least frequent) classes
/// before applying the CNC algorithm. This is particularly useful for imbalanced classification problems
/// where minority classes are of primary interest.
///
/// # Arguments
///
/// * `dataset` - The nominal dataset to analyze
/// * `n` - Number of minority classes to keep
///
/// # Returns
///
/// A `CncBpcResult` containing:
/// - The CNC result (concepts and rules)
/// - The set of minority classes that were kept
/// - Original and filtered dataset sizes
///
/// # Behavior with Ties
///
/// When classes have identical frequencies, **all classes at the same frequency level are included**.
/// The function selects complete frequency tiers until reaching or exceeding the requested `n` classes.
///
/// ## Example with Ties
///
/// Given classes A(3 objects), B(2), C(2), D(1) and `n=2`:
/// - Keeps D (most minority: 1 object)
/// - Keeps both B and C (tied at second most minority: 2 objects each)
/// - **Total: 3 classes kept** (D, B, C)
///
/// If all classes have the same frequency (complete tie), all are retained regardless of `n`.
///
/// # Parameter `n` Guide
///
/// ```text
/// cnc_bpc(&dataset, 1)  // Keep only the most minority class
/// cnc_bpc(&dataset, 2)  // Keep the 2 most minority classes (+ ties)
/// cnc_bpc(&dataset, k)  // Keep the k most minority classes (+ ties)
/// ```
///
/// # Example: Binary Classification
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc_bpc, extract_rules, display_rules};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
///
/// // Focus on the minority class
/// let result = cnc_bpc(&dataset, 1);
///
/// println!("Minority classes: {:?}", result.minority_classes);
/// println!("Kept {}/{} objects ({:.1}%)",
///          result.filtered_size,
///          result.original_size,
///          (result.filtered_size as f64 / result.original_size as f64) * 100.0);
///
/// // Extract rules focused on minority class
/// let rules = extract_rules(&dataset, &result.cnc_result);
/// display_rules(&rules);
/// # Ok(())
/// # }
/// ```
///
/// # Example: Multi-class with Multiple Minorities
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc_bpc, extract_rules, get_rules_statistics};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/contact-lenses.arff")?;
///
/// // Keep 2 most minority classes
/// let result = cnc_bpc(&dataset, 2);
///
/// println!("Original dataset: {} objects", result.original_size);
/// println!("Filtered dataset: {} objects", result.filtered_size);
/// println!("Minority classes: {:?}", result.minority_classes);
///
/// // Analyze the extracted rules
/// let rules = extract_rules(&dataset, &result.cnc_result);
/// let stats = get_rules_statistics(&rules);
/// println!("{}", stats);
/// # Ok(())
/// # }
/// ```
///
/// # Use Cases
///
/// - **Fraud detection**: Focus on rare fraudulent cases
/// - **Medical diagnosis**: Identify rare diseases
/// - **Quality control**: Detect uncommon defects
/// - **Anomaly detection**: Find unusual patterns
/// - **Multi-class imbalance**: When some classes are underrepresented
///
/// # See Also
///
/// - [`cnc`] - Standard CNC algorithm (uses all classes)
/// - [`extract_rules`] - Extract classification rules from concepts
pub fn cnc_bpc(dataset: &NominalDataset, n: usize) -> CncBpcResult {

    // We first get the G.I to not interfere on CNC.
    let pertinent_attrs = find_most_pertinent_attributes(&dataset);
    
    // Step 1: Get all class values and their distribution
    let all_class_values = dataset.get_class_values(&(0..dataset.objects.len()).collect::<Vec<_>>());
    let mut class_counts: HashMap<String, usize> = HashMap::new();
    for class_val in &all_class_values {
        *class_counts.entry(class_val.clone()).or_insert(0) += 1;
    }
    
    // Step 2: Sort classes by frequency (ascending) to find minority classes
    let mut sorted_classes: Vec<_> = class_counts.into_iter().collect();
    sorted_classes.sort_by_key(|(_, count)| *count);
    
    // Get the n most minority class names
    let minority_classes: HashSet<String> = sorted_classes.into_iter()
        .take(n)
        .map(|(class_name, _)| class_name)
        .collect();
    
    // Step 3: Create filtered dataset keeping only objects from minority classes
    let filtered_objects: Vec<usize> = dataset.data.iter().enumerate()
        .filter(|(_, obj_data)| {
            if let Some(class_val) = obj_data.get(&dataset.class_attribute) {
                minority_classes.contains(class_val)
            } else {
                false // Exclude objects with missing class
            }
        })
        .map(|(idx, _)| idx)
        .collect();
    
    // Create filtered dataset
    let filtered_objects_names: Vec<String> = filtered_objects.iter()
        .map(|&obj_idx| dataset.objects[obj_idx].clone())
        .collect();

    let filtered_data: Vec<HashMap<String, String>> = filtered_objects.iter()
        .map(|&obj_idx| dataset.data[obj_idx].clone())
        .collect();

    let filtered_dataset = NominalDataset {
        objects: filtered_objects_names,
        attributes: dataset.attributes.clone(),
        class_attribute: dataset.class_attribute.clone(),
        data: filtered_data,
    };

    // Apply CNC on filtered dataset
    let mut cnc_result = if pertinent_attrs.is_empty() {
        CncResult {
            concepts: Vec::new(),
            pertinent_attrs: Vec::new(),
        }
    } else {
        cnc_core(pertinent_attrs.clone(), &filtered_dataset)
    };

    // Map filtered indices back to original indices
    for concept in &mut cnc_result.concepts {
        concept.2 = concept.2.iter()
            .map(|&filtered_idx| filtered_objects[filtered_idx])
            .collect();
    }

    CncBpcResult {
        cnc_result,
        minority_classes,
        original_size: dataset.objects.len(),
        filtered_size: filtered_dataset.objects.len(),
    }
}

/// Extracts classification rules from CNC/CNC-BPC concepts.
///
/// This function transforms formal concepts (extent/intent pairs) into explicit
/// classification rules. Each concept generates one rule where:
/// - The **conditions** are the intent (common attributes)
/// - The **predicted class** is the majority class in the extent
/// - The **confidence** is the proportion of the majority class
///
/// # Arguments
///
/// * `dataset` - The nominal dataset used for CNC
/// * `result` - The CNC result containing concepts
///
/// # Returns
///
/// A vector of `ClassificationRule` objects, one per concept
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Load dataset and run CNC
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
///
/// // Extract rules
/// let rules = extract_rules(&dataset, &result);
///
/// println!("Extracted {} rules", rules.len());
/// for rule in &rules {
///     println!("{}", rule);
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Usage with CNC-BPC
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc_bpc, extract_rules};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc_bpc(&dataset, 1);  // Focus on minority class
///
/// // Extract rules from CNC-BPC
/// let rules = extract_rules(&dataset, &result.cnc_result);
/// # Ok(())
/// # }
/// ```
pub fn extract_rules(dataset: &NominalDataset, result: &CncResult) -> Vec<ClassificationRule> {
    let mut rules = Vec::new();
    let total_objects = dataset.objects.len();

    for (_pertinent_attr, _attr_value, extent, intent) in &result.concepts {
        if extent.is_empty() {
            continue;
        }

        // Get class distribution in extent
        let class_values = dataset.get_class_values(extent);

        if let Some((majority_class, count, _)) = NominalDataset::get_majority_class(&class_values) {
            let confidence = (count as f64 / extent.len() as f64) * 100.0;

            rules.push(ClassificationRule {
                conditions: intent.clone(),
                predicted_class: majority_class,
                confidence,
                support: extent.len(),
                total_objects,
                covered_objects: extent.clone(),
                class_attribute_name: dataset.class_attribute.clone(),
            });
        }
    }

    rules
}

/// Displays classification rules in a compact, readable format.
///
/// Each rule is displayed on one line with the format:
/// ```text
/// Rule N: IF condition1 AND condition2 ... THEN <class_attribute>=X (confidence=Y%, support=N/M, coverage=Z%)
/// ```
///
/// # Arguments
///
/// * `rules` - Slice of classification rules to display
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules, display_rules};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// // Display all rules in compact format
/// display_rules(&rules);
/// # Ok(())
/// # }
/// ```
///
/// # Output Example
///
/// ```text
/// 3 classification rule(s) extracted:
/// ================================================================================
///
/// Rule 1:
///   IF outlook=sunny THEN play=no (confidence=100.0%, support=3/14, coverage=21.4%)
///
/// Rule 2:
///   IF outlook=overcast THEN play=yes (confidence=100.0%, support=4/14, coverage=28.6%)
/// ```
pub fn display_rules(rules: &[ClassificationRule]) {
    if rules.is_empty() {
        println!("No classification rules extracted");
        return;
    }

    println!("\n{} classification rule(s) extracted:", rules.len());
    println!("{}", "=".repeat(80));

    for (i, rule) in rules.iter().enumerate() {
        println!("\nRule {}:", i + 1);
        println!("  {}", rule);
    }

    println!("\n{}", "=".repeat(80));
}

/// Displays classification rules with detailed statistics and breakdowns.
///
/// This function provides a comprehensive view of each rule including:
/// - Individual conditions listed separately
/// - Prediction with confidence
/// - Coverage statistics (support, percentage)
/// - Names of covered objects
/// - Complete class distribution in covered objects
///
/// # Arguments
///
/// * `dataset` - The dataset (needed to display object names)
/// * `rules` - Slice of classification rules to display
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules, display_rules_detailed};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// // Display detailed information for each rule
/// display_rules_detailed(&dataset, &rules);
/// # Ok(())
/// # }
/// ```
///
/// # Output Example
///
/// ```text
/// Rule 1:
///   Conditions (2 attribute(s)):
///     - outlook = sunny
///     - humidity = high
///   Prediction:
///     → class = no (confidence: 100.0%)
///   Coverage:
///     - Support: 3/14 objects (21.4%)
///     - Covered objects: ["obj1", "obj2", "obj8"]
///   Class distribution in covered objects:
///     - no: 3 (100.0%) ← predicted
/// ```
pub fn display_rules_detailed(dataset: &NominalDataset, rules: &[ClassificationRule]) {
    if rules.is_empty() {
        println!("No classification rules extracted");
        return;
    }

    println!("\n{} classification rule(s) extracted:", rules.len());
    println!("{}", "=".repeat(100));

    for (i, rule) in rules.iter().enumerate() {
        println!("\nRule {}:", i + 1);
        println!("  Conditions ({} attribute(s)):", rule.conditions.len());

        let mut conds: Vec<_> = rule.conditions.iter().collect();
        conds.sort_by_key(|(k, _)| *k);
        for (attr, value) in conds {
            println!("    - {} = {}", attr, value);
        }

        println!("  Prediction:");
        println!("    → class = {} (confidence: {:.1}%)", rule.predicted_class, rule.confidence);

        println!("  Coverage:");
        println!("    - Support: {}/{} objects ({:.1}%)",
                 rule.support, rule.total_objects, rule.coverage());

        // Show covered objects
        let covered_names: Vec<String> = rule.covered_objects.iter()
            .map(|&idx| dataset.objects[idx].clone())
            .collect();
        println!("    - Covered objects: {:?}", covered_names);

        // Show class distribution in covered objects
        let class_values = dataset.get_class_values(&rule.covered_objects);
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for class_val in &class_values {
            *class_counts.entry(class_val.clone()).or_insert(0) += 1;
        }

        println!("  Class distribution in covered objects:");
        for (class, count) in &class_counts {
            let percentage = (*count as f64 / rule.support as f64) * 100.0;
            let marker = if class == &rule.predicted_class { " ← predicted" } else { "" };
            println!("    - {}: {} ({:.1}%){}", class, count, percentage, marker);
        }
    }

    println!("\n{}", "=".repeat(100));
}

pub fn display_cnc_chosen_attribute(dataset : &NominalDataset, results : &CncResult) {

    println!("Most pertinent attribute(s): {:?}",
        results.pertinent_attrs);

    for pertinent_attr in &results.pertinent_attrs {

        let most_frequent_values = find_most_frequent_values(dataset, pertinent_attr);
        println!("  Most frequent value(s) for '{}': {:?}",
            pertinent_attr, most_frequent_values);
    }
}

/// Filters rules by minimum confidence threshold.
///
/// Returns only rules with confidence greater than or equal to the specified threshold.
/// This is useful to keep only high-quality, reliable rules.
///
/// # Arguments
///
/// * `rules` - Slice of classification rules to filter
/// * `min_confidence` - Minimum confidence threshold (0.0 to 100.0)
///
/// # Returns
///
/// A new vector containing only rules meeting the confidence threshold
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules, filter_rules_by_confidence};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// // Keep only rules with 80% or higher confidence
/// let high_quality_rules = filter_rules_by_confidence(&rules, 80.0);
///
/// println!("High quality rules: {}/{}", high_quality_rules.len(), rules.len());
/// # Ok(())
/// # }
/// ```
pub fn filter_rules_by_confidence(rules: &[ClassificationRule], min_confidence: f64) -> Vec<ClassificationRule> {
    rules.iter()
        .filter(|rule| rule.confidence >= min_confidence)
        .cloned()
        .collect()
}

/// Filters rules by minimum support threshold.
///
/// Returns only rules covering at least the specified number of objects.
/// This helps eliminate rules based on very few examples (potential noise or overfitting).
///
/// # Arguments
///
/// * `rules` - Slice of classification rules to filter
/// * `min_support` - Minimum number of objects that must be covered
///
/// # Returns
///
/// A new vector containing only rules meeting the support threshold
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules, filter_rules_by_support};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// // Keep only rules covering at least 3 objects
/// let general_rules = filter_rules_by_support(&rules, 3);
///
/// println!("General rules: {}/{}", general_rules.len(), rules.len());
/// # Ok(())
/// # }
/// ```
pub fn filter_rules_by_support(rules: &[ClassificationRule], min_support: usize) -> Vec<ClassificationRule> {
    rules.iter()
        .filter(|rule| rule.support >= min_support)
        .cloned()
        .collect()
}

/// Sorts rules by confidence in descending order (highest confidence first).
///
/// Modifies the input slice in-place to order rules from most confident to least confident.
///
/// # Arguments
///
/// * `rules` - Mutable slice of classification rules to sort
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules, sort_rules_by_confidence};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let mut rules = extract_rules(&dataset, &result);
///
/// // Sort by confidence
/// sort_rules_by_confidence(&mut rules);
///
/// // Display top 3 rules
/// for (i, rule) in rules.iter().take(3).enumerate() {
///     println!("{}. {} (conf: {:.1}%)", i+1, rule.predicted_class, rule.confidence);
/// }
/// # Ok(())
/// # }
/// ```
pub fn sort_rules_by_confidence(rules: &mut [ClassificationRule]) {
    rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
}

/// Sorts rules by support in descending order (highest support first).
///
/// Modifies the input slice in-place to order rules from most general (covering more objects)
/// to most specific (covering fewer objects).
///
/// # Arguments
///
/// * `rules` - Mutable slice of classification rules to sort
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules, sort_rules_by_support};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let mut rules = extract_rules(&dataset, &result);
///
/// // Sort by support to find most general rules
/// sort_rules_by_support(&mut rules);
///
/// // Display top 3 most general rules
/// for (i, rule) in rules.iter().take(3).enumerate() {
///     println!("{}. Support: {}, Coverage: {:.1}%", i+1, rule.support, rule.coverage());
/// }
/// # Ok(())
/// # }
/// ```
pub fn sort_rules_by_support(rules: &mut [ClassificationRule]) {
    rules.sort_by(|a, b| b.support.cmp(&a.support));
}

/// Computes summary statistics for a set of classification rules.
///
/// Returns a `RulesStatistics` object containing aggregate metrics across all rules:
/// - Total number of rules
/// - Average, minimum, and maximum confidence
/// - Average, minimum, and maximum support
/// - Average number of conditions per rule
/// - Number of unique predicted classes
///
/// # Arguments
///
/// * `rules` - Slice of classification rules to analyze
///
/// # Returns
///
/// A `RulesStatistics` object with computed metrics
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules, get_rules_statistics};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// let stats = get_rules_statistics(&rules);
/// println!("{}", stats);
/// // Displays:
/// // Rules Statistics:
/// //   Total rules: 8
/// //   Confidence: avg=95.5%, min=80.0%, max=100.0%
/// //   Support: avg=2.5, min=1, max=5
/// //   ...
/// # Ok(())
/// # }
/// ```
pub fn get_rules_statistics(rules: &[ClassificationRule]) -> RulesStatistics {
    if rules.is_empty() {
        return RulesStatistics::default();
    }

    let avg_confidence = rules.iter().map(|r| r.confidence).sum::<f64>() / rules.len() as f64;
    let avg_support = rules.iter().map(|r| r.support).sum::<usize>() as f64 / rules.len() as f64;
    let avg_conditions = rules.iter().map(|r| r.conditions.len()).sum::<usize>() as f64 / rules.len() as f64;

    let max_confidence = rules.iter().map(|r| r.confidence).fold(0.0, f64::max);
    let min_confidence = rules.iter().map(|r| r.confidence).fold(100.0, f64::min);

    let max_support = rules.iter().map(|r| r.support).max().unwrap_or(0);
    let min_support = rules.iter().map(|r| r.support).min().unwrap_or(0);

    // Count unique predicted classes
    let unique_classes: HashSet<String> = rules.iter()
        .map(|r| r.predicted_class.clone())
        .collect();

    RulesStatistics {
        total_rules: rules.len(),
        avg_confidence,
        min_confidence,
        max_confidence,
        avg_support,
        min_support,
        max_support,
        avg_conditions,
        unique_predicted_classes: unique_classes.len(),
    }
}

/// Summary statistics computed over a set of classification rules.
///
/// This structure aggregates metrics across multiple rules to provide
/// an overview of the rule set quality and characteristics.
///
/// # Fields
///
/// - `total_rules`: Total number of rules
/// - `avg_confidence`: Average confidence across all rules
/// - `min_confidence`: Minimum confidence value
/// - `max_confidence`: Maximum confidence value
/// - `avg_support`: Average number of objects covered per rule
/// - `min_support`: Minimum support value
/// - `max_support`: Maximum support value
/// - `avg_conditions`: Average number of conditions (attributes) per rule
/// - `unique_predicted_classes`: Number of distinct classes predicted by the rules
///
/// # Example
///
/// ```
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules, get_rules_statistics};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// let stats = get_rules_statistics(&rules);
///
/// // Access individual statistics
/// if stats.avg_confidence > 90.0 {
///     println!("High quality rule set!");
/// }
///
/// if stats.unique_predicted_classes == 2 {
///     println!("Binary classification problem");
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct RulesStatistics {
    pub total_rules: usize,
    pub avg_confidence: f64,
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub avg_support: f64,
    pub min_support: usize,
    pub max_support: usize,
    pub avg_conditions: f64,
    pub unique_predicted_classes: usize,
}

impl std::fmt::Display for RulesStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Rules Statistics:")?;
        writeln!(f, "  Total rules: {}", self.total_rules)?;
        writeln!(f, "  Confidence: avg={:.1}%, min={:.1}%, max={:.1}%",
                 self.avg_confidence, self.min_confidence, self.max_confidence)?;
        writeln!(f, "  Support: avg={:.1}, min={}, max={}",
                 self.avg_support, self.min_support, self.max_support)?;
        writeln!(f, "  Avg conditions per rule: {:.1}", self.avg_conditions)?;
        writeln!(f, "  Unique predicted classes: {}", self.unique_predicted_classes)?;
        Ok(())
    }
}

/// Display CNC results in a standardized format
pub fn display_cnc_results(dataset: &NominalDataset, results: &[(String, String, Vec<usize>, HashMap<String, String>)]) {
    if results.is_empty() {
        println!("No concepts found");
        return;
    }
    
    // There is a theroem ("connexion de Galois") saying that A''' = A'.
    println!("\n{} concept(s) found:", results.len());
    
    for (i, (pertinent_attr, attr_value, extent, intent)) in results.iter().enumerate() {
        println!("\nConcept {}:", i + 1);
        println!("  Pertinent attribute: '{}' with value '{}'", pertinent_attr, attr_value);
        
        // Show extent (objects)
        let extent_objects: Vec<String> = extent.iter()
            .map(|&obj_idx| dataset.objects[obj_idx].clone())
            .collect();
        println!("  Extent of the pertinent attribute(s): {:?}", extent_objects);
        println!("  Extent size: {}/{} objects ({:.1}%)", 
                 extent.len(), dataset.objects.len(), 
                 (extent.len() as f64 / dataset.objects.len() as f64) * 100.0);
        
        // Show intent (common attributes)
        println!("  Intent (common attributes) of the found extent : {:?}", intent);
        let desc_attrs_count = dataset.attributes.iter().filter(|a| *a != &dataset.class_attribute).count();
        println!("  Intent size: {}/{} attributes ({:.1}%)", 
                 intent.len(), desc_attrs_count,
                 (intent.len() as f64 / desc_attrs_count as f64) * 100.0);
        
        // Show class distribution in extent
        let class_values = dataset.get_class_values(&extent);
        let majority_class = NominalDataset::get_majority_class(&class_values)
            .map(|(class, _, _)| class);

        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for class_val in &class_values {
            *class_counts.entry(class_val.clone()).or_insert(0) += 1;
        }

        println!("  Class distribution in extent:");
        for (class_val, count) in &class_counts {
            let percentage = (*count as f64 / extent.len() as f64) * 100.0;
            let majority_marker = if Some(class_val.clone()) == majority_class {
                " (majority class)"
            } else {
                ""
            };
            println!("    {}: {} ({:.1}%){}", class_val, count, percentage, majority_marker);
        }
    }
}

/// Loads an ARFF file and converts it to a NominalDataset for CNC/CNC-BPC.
///
/// This function parses ARFF (Attribute-Relation File Format) files and creates
/// a `NominalDataset` suitable for use with CNC and CNC-BPC algorithms.
///
/// # Arguments
///
/// * `file_path` - Path to the .arff file
/// * `class_attr_name` - Name of the class attribute (must match exactly the attribute name in the ARFF file)
///
/// # Returns
///
/// A `Result` containing the `NominalDataset` or an error
///
/// # Important Notes
///
/// ## Nominal vs Numeric Attributes
///
/// CNC and CNC-BPC work on **nominal (categorical)** data. If your ARFF file contains numeric attributes:
///
/// 1. **Recommended**: Discretize them first (convert to categories) for better results
/// 2. **Quick option**: The parser converts them to strings, but results may be less meaningful
///
/// ## Finding the Class Attribute Name
///
/// The class attribute name must match exactly. Common patterns:
/// - `weather.nominal.arff` → `"play"`
/// - `contact-lenses.arff` → `"contact-lenses"`
/// - `iris.arff` → `"class"`
///
/// Find it by looking at the last `@attribute` line in the ARFF file.
///
/// ## Recommended Datasets
///
/// **Nominal datasets** (ready to use):
/// - `weather.nominal.arff` - 14 objects, 5 attributes, 2 classes
/// - `contact-lenses.arff` - 24 objects, 5 attributes, 3 classes
/// - `vote.arff` - Congressional voting records
/// - `labor.arff` - Labor negotiations
///
/// **Numeric datasets** (require discretization for best results):
/// - `iris.arff` - 4 numeric attributes
/// - `diabetes.arff` - Numeric health data
/// - `cpu.arff` - Computer performance
///
/// # Example
///
/// ```no_run
/// use fcars::cnc::{from_arff, cnc, extract_rules, display_rules};
///
/// // Load the dataset
/// let dataset = from_arff("data-examples/weather.nominal.arff", "play")?;
///
/// // Display summary
/// dataset.display_summary();
///
/// // Run CNC
/// let result = cnc(&dataset);
///
/// // Extract and display rules
/// let rules = extract_rules(&dataset, &result);
/// display_rules(&rules);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example with CNC-BPC
///
/// ```no_run
/// use fcars::cnc::{from_arff, cnc_bpc};
///
/// let dataset = from_arff("data-examples/contact-lenses.arff", "contact-lenses")?;
///
/// // Focus on minority classes
/// let result = cnc_bpc(&dataset, 1);
/// println!("Minority classes kept: {:?}", result.minority_classes);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be read
/// - File is not valid ARFF format
/// - Class attribute name not found in the file
pub fn from_arff(
    file_path: &str,
    class_attr_name: &str,
) -> Result<NominalDataset, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;

    let mut attributes = Vec::new();
    let mut in_data_section = false;
    let mut objects = Vec::new();
    let mut data = Vec::new();
    let mut obj_counter = 0;

    for line in content.lines() {
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        // Check if we're in the data section
        if line.to_lowercase().starts_with("@data") {
            in_data_section = true;
            continue;
        }

        if !in_data_section {
            // Parse attribute declarations
            if line.to_lowercase().starts_with("@attribute") {
                // Format: @attribute name type
                // We just need the name
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let attr_name = parts[1].trim()
                        .trim_matches('\'')  // Remove single quotes
                        .trim_matches('"');  // Remove double quotes
                    attributes.push(attr_name.to_string());
                }
            }
        } else {
            // Parse data lines (CSV format)
            let values: Vec<String> = line
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();

            if values.len() == attributes.len() {
                objects.push(format!("obj_{}", obj_counter));
                obj_counter += 1;

                let mut obj_data = HashMap::new();
                for (i, value) in values.iter().enumerate() {
                    obj_data.insert(attributes[i].clone(), value.clone());
                }
                data.push(obj_data);
            }
        }
    }

    // Verify class attribute exists
    if !attributes.contains(&class_attr_name.to_string()) {
        return Err(format!(
            "Class attribute '{}' not found. Available: {:?}",
            class_attr_name, attributes
        )
        .into());
    }

    Ok(NominalDataset::new(
        objects,
        attributes,
        class_attr_name.to_string(),
        data,
    ))
}

/// Loads an ARFF file using the last attribute as class (ARFF convention).
///
/// This is a convenience function that automatically uses the last attribute
/// in the ARFF file as the class attribute, following the standard ARFF convention
/// where the target variable is typically the last column.
///
/// # Arguments
///
/// * `file_path` - Path to the .arff file
///
/// # Returns
///
/// A `Result` containing the `NominalDataset` or an error
///
/// # When to Use
///
/// Use this function when:
/// - Your ARFF file follows the standard convention (class is last attribute)
/// - You want quick loading without specifying the class attribute name
/// - You're working with standard benchmark datasets
///
/// If the class attribute is **not** the last one, use [`from_arff`] instead and specify the class name explicitly.
///
/// # Example: Quick Start
///
/// ```no_run
/// use fcars::cnc::{from_arff_auto, cnc, extract_rules};
///
/// // Automatically uses last attribute as class
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
///
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// println!("Found {} rules", rules.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: Complete Workflow
///
/// ```no_run
/// use fcars::cnc::{from_arff_auto, cnc_bpc, extract_rules, display_rules, get_rules_statistics};
///
/// // Load dataset (uses last attribute as class)
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
///
/// // Run CNC-BPC with minority class focus
/// let result = cnc_bpc(&dataset, 1);
/// println!("Filtered to {} objects (from {})",
///          result.filtered_size, result.original_size);
///
/// // Extract and analyze rules
/// let rules = extract_rules(&dataset, &result.cnc_result);
/// display_rules(&rules);
///
/// let stats = get_rules_statistics(&rules);
/// println!("\n{}", stats);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # See Also
///
/// - [`from_arff`] - Load ARFF with explicit class attribute name
/// - [`cnc`] - Run the CNC algorithm
/// - [`cnc_bpc`] - Run CNC-BPC for imbalanced data
pub fn from_arff_auto(
    file_path: &str,
) -> Result<NominalDataset, Box<dyn std::error::Error>> {
    // First, we need to read the file to get the last attribute
    let content = fs::read_to_string(file_path)?;

    let mut attributes = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        if line.to_lowercase().starts_with("@data") {
            break;
        }
        if line.to_lowercase().starts_with("@attribute") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let attr_name = parts[1].trim()
                    .trim_matches('\'')  // Remove single quotes
                    .trim_matches('"');  // Remove double quotes
                attributes.push(attr_name.to_string());
            }
        }
    }

    if attributes.is_empty() {
        return Err("No attributes found in ARFF file".into());
    }

    let class_attr = attributes.last().unwrap();
    from_arff(file_path, class_attr)
}
