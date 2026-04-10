//! Core CNC (Classifier Nominal Concept) algorithm implementation.
//!
//! This module provides the core CNC and CNC-BPC algorithms for classification of nominal
//! (categorical) data using Formal Concept Analysis, along with dataset structures and
//! ARFF file loading capabilities.

use std::collections::{HashMap, HashSet};
use std::fs;

/// Result structure for CNC containing both the concepts and debug information
#[derive(Debug)]
pub struct CncResult {
    /// List of concepts: (attribute_name, attribute_value, extent, intent)
    /// - attribute_name: The pertinent attribute
    /// - attribute_value: The most frequent value
    /// - extent: Indices of objects in the concept
    /// - intent: Common attribute-value pairs across all objects
    pub concepts: Vec<(String, String, Vec<usize>, HashMap<String, String>)>,
    /// List of pertinent attributes used in the analysis
    pub pertinent_attrs: Vec<String>,
}

/// Result structure for CNC-BPC containing both the concepts and debug information.
/// Uses CncResult to avoid duplication and maintain consistency
#[derive(Debug)]
pub struct CncBpcResult {
    /// The CNC result computed on the filtered dataset
    pub cnc_result: CncResult,
    /// Set of minority classes that were kept in the filtered dataset
    pub minority_classes: HashSet<String>,
    /// Number of objects in the original dataset
    pub original_size: usize,
    /// Number of objects in the filtered dataset
    pub filtered_size: usize,
}

/// Represents a nominal (categorical) dataset for classification using CNC/CNC-BPC.
///
/// A `NominalDataset` stores categorical data where both attributes and class values
/// are represented as strings. This structure is designed for Formal Concept Analysis
/// and works exclusively with nominal (non-numeric) data.
///
/// # Structure
///
/// - **Objects**: Individual instances/samples in the dataset (e.g., "patient1", "obj42")
/// - **Attributes**: Features describing the objects (e.g., "color", "size", "temperature")
/// - **Class attribute**: The target variable for classification (e.g., "diagnosis", "species")
/// - **Data**: Attribute-value pairs for each object, stored as a vector of HashMaps
///
/// # Important Notes
///
/// - All values are **strings** (nominal/categorical data)
/// - The class attribute is included in the `attributes` list but treated specially
/// - Each object has one HashMap containing all its attribute-value pairs
/// - Missing values should be handled before creating the dataset
///
/// # Example
///
/// ```
/// use cnc::NominalDataset;
/// use std::collections::HashMap;
///
/// let objects = vec!["patient1".to_string(), "patient2".to_string()];
/// let attributes = vec!["fever".to_string(), "cough".to_string(), "diagnosis".to_string()];
/// let class_attribute = "diagnosis".to_string();
///
/// let mut patient1_data = HashMap::new();
/// patient1_data.insert("fever".to_string(), "high".to_string());
/// patient1_data.insert("cough".to_string(), "yes".to_string());
/// patient1_data.insert("diagnosis".to_string(), "flu".to_string());
///
/// let mut patient2_data = HashMap::new();
/// patient2_data.insert("fever".to_string(), "none".to_string());
/// patient2_data.insert("cough".to_string(), "no".to_string());
/// patient2_data.insert("diagnosis".to_string(), "healthy".to_string());
///
/// let data = vec![patient1_data, patient2_data];
///
/// let dataset = NominalDataset::new(objects, attributes, class_attribute, data);
/// assert_eq!(dataset.objects.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct NominalDataset {
    /// Names/identifiers of objects (samples/instances) in the dataset
    pub objects: Vec<String>,
    /// Names of all attributes, including both descriptive attributes and the class attribute
    pub attributes: Vec<String>,
    /// Name of the class attribute (target variable for classification)
    pub class_attribute: String,
    /// Attribute-value pairs for each object (one HashMap per object, indexed by object position)
    pub data: Vec<HashMap<String, String>>,
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
    
    /// Get all possible values for an attribute (sorted alphabetically)
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
    ///
    /// Returns a HashMap mapping attribute values to vectors of object indices.
    pub fn group_by_attribute_value(&self, attr_name: &str) -> HashMap<String, Vec<usize>> {
        let mut groups = HashMap::new();
        for (obj_idx, obj_data) in self.data.iter().enumerate() {
            if let Some(val) = obj_data.get(attr_name) {
                groups.entry(val.clone()).or_insert_with(Vec::new).push(obj_idx);
            }
        }
        groups
    }
    
    /// Extracts the class attribute values for a specific subset of objects.
    ///
    /// Given a list of object indices, this function retrieves the corresponding class
    /// values in the same order as the indices. Objects without a class value are skipped.
    ///
    /// # Arguments
    ///
    /// * `object_indices` - Slice of object indices (positions in the dataset)
    ///
    /// # Returns
    ///
    /// A vector of class values (as strings) in the same order as `object_indices`.
    /// If an object doesn't have a class value, it's omitted from the result.
    ///
    /// # Example
    ///
    /// ```
    /// use cnc::NominalDataset;
    /// use std::collections::HashMap;
    ///
    /// let objects = vec!["o1".to_string(), "o2".to_string(), "o3".to_string()];
    /// let attributes = vec!["attr".to_string(), "class".to_string()];
    /// let mut data = vec![];
    ///
    /// for class_val in ["A", "B", "A"] {
    ///     let mut obj = HashMap::new();
    ///     obj.insert("attr".to_string(), "x".to_string());
    ///     obj.insert("class".to_string(), class_val.to_string());
    ///     data.push(obj);
    /// }
    ///
    /// let dataset = NominalDataset::new(objects, attributes, "class".to_string(), data);
    ///
    /// // Get class values for objects at indices 0 and 2
    /// let classes = dataset.get_class_values(&[0, 2]);
    /// assert_eq!(classes, vec!["A".to_string(), "A".to_string()]);
    /// ```
    pub fn get_class_values(&self, object_indices: &[usize]) -> Vec<String> {
        object_indices.iter()
            .filter_map(|&obj_idx| self.data[obj_idx].get(&self.class_attribute).cloned())
            .collect()
    }
    
    /// Finds the most frequent class in a collection of class values.
    ///
    /// This static method analyzes a slice of class labels and determines which class
    /// appears most often, along with its count and percentage of the total.
    ///
    /// # Arguments
    ///
    /// * `class_values` - Slice of class labels (can contain duplicates)
    ///
    /// # Returns
    ///
    /// Returns `Some((majority_class, count, percentage))` where:
    /// - `majority_class`: The most frequent class label
    /// - `count`: Number of times it appears
    /// - `percentage`: Proportion of total (0.0 to 100.0)
    ///
    /// Returns `None` if the input is empty.
    ///
    /// # Note
    ///
    /// In case of a tie (multiple classes with the same maximum count), the behavior
    /// is non-deterministic and depends on HashMap iteration order.
    ///
    /// # Example
    ///
    /// ```
    /// use cnc::NominalDataset;
    ///
    /// let classes = vec!["A".to_string(), "A".to_string(), "B".to_string()];
    /// let (majority, count, pct) = NominalDataset::get_majority_class(&classes).unwrap();
    ///
    /// assert_eq!(majority, "A");
    /// assert_eq!(count, 2);
    /// assert_eq!(pct, 66.66666666666666);
    /// ```
    ///
    /// ```
    /// use cnc::NominalDataset;
    ///
    /// // Empty input returns None
    /// let classes: Vec<String> = vec![];
    /// assert!(NominalDataset::get_majority_class(&classes).is_none());
    /// ```
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
    
    /// Displays only the dataset context (formatted table).
    ///
    /// This method prints the complete dataset as a formatted table showing all objects
    /// and their attribute values, with the class attribute visually separated on the right.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cnc::{from_arff_auto};
    ///
    /// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
    /// dataset.display_context();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Output Example
    ///
    /// ```text
    /// Context:
    ///                    fever          cough       │  diagnosis
    ///                                               │
    ///   patient1          high            yes       │        flu
    ///   patient2          none             no       │    healthy
    ///   patient3          mild            yes       │       cold
    /// ```
    pub fn display_context(&self) {
        println!("Context:\n{}", &self);
    }

    /// Displays only the dataset summary statistics.
    ///
    /// This method prints:
    /// - Number of objects (samples/instances)
    /// - Number and names of descriptive attributes
    /// - Class attribute name
    /// - Number of all possible values for each descriptive attribute
    /// - Class distribution (count and percentage for each class)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cnc::{from_arff_auto};
    ///
    /// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
    /// dataset.display_dataset_summary();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Output Example
    ///
    /// ```text
    /// Dataset Summary:
    /// - Objects: 3
    /// - Descriptive attributes: 2
    /// - Class attribute: diagnosis
    /// - Attribute 'fever': no more than 3 possible values
    /// - Attribute 'cough': no more than 2 possible values
    /// - Class distribution:
    ///   cold: 1 (33.3%)
    ///   flu: 1 (33.3%)
    ///   healthy: 1 (33.3%)
    /// ```
    pub fn display_dataset_summary(&self) {
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
            println!("- Attribute '{}': no more than {} possible values", attr, values.len());
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

    /// Displays comprehensive summary about the dataset with smart selection.
    ///
    /// This method intelligently chooses what to display based on dataset size:
    /// - **Small datasets** (≤ 15 objects with ≤ 7 possible values per attribute with ≤ 7 attributes): shows only the context table
    /// - **All other datasets**: shows only the summary statistics
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cnc::{from_arff_auto};
    ///
    /// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
    /// dataset.display_summary();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`display_context`] - Display only the formatted table
    /// - [`display_dataset_summary`] - Display only the statistics
    pub fn display_summary(&self) {
        let (show_context, show_summary) = self.determine_display_mode();

        if show_context {
            self.display_context();
        } else if show_summary {
            self.display_dataset_summary();
        }
    }

    /// Determines what to display based on dataset characteristics.
    ///
    /// Returns a tuple `(show_context, show_summary)`:
    /// - Small datasets: `(true, false)` - Context is readable and sufficient
    /// - All other datasets: `(false, true)` - Summary is more appropriate
    ///
    /// A dataset is considered "small" if ALL of these conditions are met:
    /// - Number of objects ≤ 15
    /// - Maximum unique values per attribute ≤ 7
    /// - Number of descriptive attributes ≤ 7
    fn determine_display_mode(&self) -> (bool, bool) {
        let num_objects = self.objects.len();

        // Count descriptive attributes (excluding class)
        let desc_attrs: Vec<_> = self.attributes.iter()
            .filter(|attr| attr != &&self.class_attribute)
            .collect();

        let num_attributes = desc_attrs.len();

        // Find maximum number of unique values across all descriptive attributes
        let max_unique_values: usize = desc_attrs.iter()
            .map(|attr| self.get_attribute_values(attr).len())
            .max()
            .unwrap_or(0);

        // Decision thresholds
        const MAX_OBJECTS_FOR_CONTEXT: usize = 15;
        const MAX_UNIQUE_VALUES_FOR_CONTEXT: usize = 7;
        const MAX_ATTRIBUTES_FOR_CONTEXT: usize = 7;

        // Small dataset: context is readable and sufficient
        if num_objects <= MAX_OBJECTS_FOR_CONTEXT
            && max_unique_values <= MAX_UNIQUE_VALUES_FOR_CONTEXT
            && num_attributes <= MAX_ATTRIBUTES_FOR_CONTEXT
        {
            return (true, false);
        }

        // All other cases: summary is more appropriate
        (false, true)
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
///
/// This function computes the formal concept closure for a given attribute-value pair.
/// It returns the extent (object indices) and intent (common attribute-value pairs).
///
/// # Arguments
///
/// * `dataset` - The nominal dataset
/// * `attr_name` - The attribute name to use as starting point
/// * `attr_value` - The attribute value to match
///
/// # Returns
///
/// A tuple of (extent, intent):
/// - `extent`: Vector of object indices that have the specified attribute-value
/// - `intent`: HashMap of attribute-value pairs common to all objects in the extent
///
/// # Example
///
/// ```
/// use cnc::{NominalDataset, compute_nominal_closure};
/// use std::collections::HashMap;
///
/// let dataset = NominalDataset::new(
///     vec!["obj1".into(), "obj2".into(), "obj3".into()],
///     vec!["color".into(), "size".into(), "class".into()],
///     "class".into(),
///     vec![
///         {
///             let mut m = HashMap::new();
///             m.insert("color".into(), "red".into());
///             m.insert("size".into(), "big".into());
///             m.insert("class".into(), "A".into());
///             m
///         },
///         {
///             let mut m = HashMap::new();
///             m.insert("color".into(), "red".into());
///             m.insert("size".into(), "big".into());
///             m.insert("class".into(), "A".into());
///             m
///         },
///         {
///             let mut m = HashMap::new();
///             m.insert("color".into(), "blue".into());
///             m.insert("size".into(), "small".into());
///             m.insert("class".into(), "B".into());
///             m
///         },
///     ],
/// );
///
/// // Find all objects with color=red and their common attributes
/// let (extent, intent) = compute_nominal_closure(&dataset, "color", "red");
///
/// assert_eq!(extent, vec![0, 1]); // obj1 and obj2
/// assert_eq!(intent.get("color"), Some(&"red".to_string()));
/// assert_eq!(intent.get("size"), Some(&"big".to_string()));
/// ```
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

/// Runs the CNC (Classifier Nominal Concept) algorithm on a nominal dataset.
///
/// The CNC algorithm finds the most pertinent attribute(s) using information gain,
/// identifies the most frequent value(s) for each pertinent attribute, and computes
/// formal concepts (extent/intent pairs) that can be used for classification.
///
/// # Algorithm Steps
///
/// 1. **Find pertinent attributes**: Selects attribute(s) with maximum information gain
/// 2. **Find frequent values**: For each pertinent attribute, finds the most frequent value(s)
/// 3. **Compute concepts**: Calculates the closure (extent and intent) for each attribute-value pair
///
/// # Arguments
///
/// * `dataset` - The nominal dataset to analyze
///
/// # Returns
///
/// A `CncResult` containing:
/// - `concepts`: Vector of tuples (attribute, value, extent, intent)
/// - `pertinent_attrs`: List of attributes selected as most pertinent
///
/// # Ties Handling
///
/// When multiple attributes have the same information gain, or multiple values have
/// the same frequency, all tied options are included in the result.
///
/// # Example
///
/// ```
/// use cnc::{from_arff_auto, cnc};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
///
/// println!("Found {} concepts", result.concepts.len());
/// println!("Pertinent attributes: {:?}", result.pertinent_attrs);
///
/// for (attr, value, extent, intent) in &result.concepts {
///     println!("Concept: {}={} covers {} objects", attr, value, extent.len());
/// }
/// # Ok(())
/// # }
/// ```
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
/// use cnc::{from_arff_auto, cnc_bpc};
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
/// # Ok(())
/// # }
/// ```
///
/// # Example: Multi-class with Multiple Minorities
///
/// ```
/// use cnc::{from_arff_auto, cnc_bpc};
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

/// Displays the pertinent attribute(s) and their most frequent values from CNC results.
///
/// This function provides a quick summary of which attributes were selected as most
/// pertinent by the CNC algorithm and which values were most frequent for each.
///
/// # Arguments
///
/// * `dataset` - The dataset used for analysis
/// * `results` - The CNC results containing pertinent attributes
///
/// # Output Format
///
/// ```text
/// Most pertinent attribute(s): ["Outlook"]
///   Most frequent value(s) for 'Outlook': ["Sunny"]
/// ```
///
/// # Example
///
/// ```
/// use cnc::{from_arff_auto, cnc, display_cnc_chosen_attribute};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
///
/// display_cnc_chosen_attribute(&dataset, &result);
/// # Ok(())
/// # }
/// ```
pub fn display_cnc_chosen_attribute(dataset : &NominalDataset, results : &CncResult) {

    println!("Most pertinent attribute(s): {:?}",
        results.pertinent_attrs);

    for pertinent_attr in &results.pertinent_attrs {

        let most_frequent_values = find_most_frequent_values(dataset, pertinent_attr);
        println!("  Most frequent value(s) for '{}': {:?}",
            pertinent_attr, most_frequent_values);
    }
}

/// Internal function to display CNC results.
///
/// This private function implements the core display logic and is called by the public
/// `display_cnc_results_consistently` and `display_cnc_results_inconsistently` functions.
///
/// # Arguments
///
/// * `dataset` - The dataset (needed for object names and class information)
/// * `results` - Vector of concepts (pertinent_attr, attr_value, extent, intent)
/// * `sort_output` - If true, sorts Intent attributes and class distribution alphabetically
fn display_cnc_results(dataset: &NominalDataset, results: &[(String, String, Vec<usize>, HashMap<String, String>)], sort_output: bool) {
    if results.is_empty() {
        println!("No concepts found");
        return;
    }

    // There is a theorem ("connexion de Galois") saying that A''' = A'.
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
        if sort_output {
            // Sorted version for deterministic output
            let mut sorted_intent: Vec<_> = intent.iter().collect();
            sorted_intent.sort_by_key(|(k, _)| *k);
            print!("  Intent (common attributes) of the found extent : {{");
            for (idx, (attr, value)) in sorted_intent.iter().enumerate() {
                if idx > 0 {
                    print!(", ");
                }
                print!("\"{}\": \"{}\"", attr, value);
            }
            println!("}}");
        } else {
            // Fast version with no sorting
            println!("  Intent (common attributes) of the found extent : {:?}", intent);
        }

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

        if sort_output {
            // Sorted version for deterministic output
            let mut sorted_classes: Vec<_> = class_counts.iter().collect();
            sorted_classes.sort_by_key(|(k, _)| *k);
            for (class_val, count) in sorted_classes {
                let percentage = (*count as f64 / extent.len() as f64) * 100.0;
                if Some(class_val) == majority_class.as_ref() {
                    println!("    {}: {} ({:.1}%) (majority class)", class_val, count, percentage);
                } else {
                    println!("    {}: {} ({:.1}%)", class_val, count, percentage);
                }
            }
        } else {
            // Fast version with no sorting
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
}

/// Display CNC/CNC-BPC results with **consistent, deterministic output**.
///
/// This function displays the concepts found by CNC/CNC-BPC with:
/// - Intent attributes sorted alphabetically by attribute name
/// - Class distribution sorted alphabetically by class name
///
/// This ensures **deterministic and reproducible output**, which is useful for:
/// - Automated testing
/// - Generating consistent documentation
/// - Comparing results across runs
///
/// # Performance Note
///
/// This function sorts the Intent attributes and class distribution for each concept.
/// If you don't need deterministic output and want optimal performance, use
/// [`display_cnc_results_inconsistently`] instead.
///
/// # Arguments
///
/// * `dataset` - The nominal dataset
/// * `results` - Vector of concepts (pertinent_attr, attr_value, extent, intent)
///
/// # Example
///
/// ```
/// use cnc::{from_arff_auto, cnc, display_cnc_results_consistently};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
///
/// // Deterministic display (for tests and documentation)
/// display_cnc_results_consistently(&dataset, &result.concepts);
/// # Ok(())
/// # }
/// ```
///
/// # Output Format
///
/// ```text
/// Concept 1:
///   Pertinent attribute: 'Windy' with value 'False'
///   Extent of the pertinent attribute(s): ["o9", "o10", "o13"]
///   Extent size: 3/5 objects (60.0%)
///   Intent (common attributes) of the found extent : {"Humidity": "Normal", "Windy": "False"}
///   Intent size: 2/4 attributes (50.0%)
///   Class distribution in extent:
///     Yes: 3 (100.0%) (majority class)
/// ```
///
/// Note: Intent attributes are sorted alphabetically (Humidity before Windy).
pub fn display_cnc_results_consistently(dataset: &NominalDataset, results: &[(String, String, Vec<usize>, HashMap<String, String>)]) {
    display_cnc_results(dataset, results, true);
}

/// Display CNC/CNC-BPC results with **inconsistent, optimized output**.
///
/// This function displays the concepts found by CNC/CNC-BPC with:
/// - Intent attributes in HashMap iteration order (non-deterministic)
/// - Class distribution in HashMap iteration order (non-deterministic)
///
/// This provides **optimal performance** (no sorting overhead) but:
/// - Output order may vary between runs
/// - Not suitable for automated testing or result comparison
///
/// # Performance Note
///
/// This function is the fastest option as it avoids sorting. If you need
/// deterministic output, use [`display_cnc_results_consistently`] instead.
///
/// # Arguments
///
/// * `dataset` - The nominal dataset
/// * `results` - Vector of concepts (pertinent_attr, attr_value, extent, intent)
///
/// # Example
///
/// ```
/// use cnc::{from_arff_auto, cnc, display_cnc_results_inconsistently};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
///
/// // Fast display (non-deterministic order)
/// display_cnc_results_inconsistently(&dataset, &result.concepts);
/// # Ok(())
/// # }
/// ```
///
/// # Output Format
///
/// ```text
/// Concept 1:
///   Pertinent attribute: 'Windy' with value 'False'
///   Extent of the pertinent attribute(s): ["o9", "o10", "o13"]
///   Extent size: 3/5 objects (60.0%)
///   Intent (common attributes) of the found extent : {"Windy": "False", "Humidity": "Normal"}
///   Intent size: 2/4 attributes (50.0%)
///   Class distribution in extent:
///     Yes: 3 (100.0%) (majority class)
/// ```
///
/// Note: Intent attribute order is non-deterministic and may vary between runs.
pub fn display_cnc_results_inconsistently(dataset: &NominalDataset, results: &[(String, String, Vec<usize>, HashMap<String, String>)]) {
    display_cnc_results(dataset, results, false);
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
/// use cnc::{from_arff, cnc, display_cnc_results_consistently};
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
/// // Display results
/// display_cnc_results_consistently(&dataset, &result.concepts);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example with CNC-BPC
///
/// ```no_run
/// use cnc::{from_arff, cnc_bpc};
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
/// use cnc::{from_arff_auto, cnc, display_cnc_results_consistently};
///
/// // Automatically uses last attribute as class
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
///
/// let result = cnc(&dataset);
/// display_cnc_results_consistently(&dataset, &result.concepts);
///
/// println!("Found {} concepts", result.concepts.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: Complete Workflow
///
/// ```no_run
/// use cnc::{from_arff_auto, cnc_bpc, display_cnc_results_consistently};
///
/// // Load dataset (uses last attribute as class)
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
///
/// // Run CNC-BPC with minority class focus
/// let result = cnc_bpc(&dataset, 1);
/// println!("Filtered to {} objects (from {})",
///          result.filtered_size, result.original_size);
///
/// // Display results
/// display_cnc_results_consistently(&dataset, &result.cnc_result.concepts);
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
