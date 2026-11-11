//! `ModelEvaluator::information_item_type`: what a `typeRef` resolves to, as the
//! evaluators resolve it.

use crate::model_evaluator::ModelEvaluator;
use dsntk_feel::FeelType;

const NAMESPACE: &str = "https://dsntk.io/type-refs";

const MODEL: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" namespace="https://dsntk.io/type-refs" name="type-refs" id="_type-refs">
  <itemDefinition name="Tier"><typeRef>string</typeRef></itemDefinition>
  <inputData name="Customer" id="customer"><variable name="Customer" typeRef="Tier"/></inputData>
</definitions>"#;

#[test]
fn _0001() {
  let definitions = dsntk_model::parse(MODEL).unwrap();
  let evaluator = ModelEvaluator::new(&[definitions]).unwrap();
  let resolve = |type_ref: &str| evaluator.information_item_type(NAMESPACE, type_ref);
  assert_eq!(Some(FeelType::String), resolve("string"));
  assert_eq!(Some(FeelType::Number), resolve("number"));
  assert_eq!(Some(FeelType::Any), resolve("Any"));
  assert_eq!(Some(FeelType::DateTime), resolve("dateTime"));
  assert_eq!(Some(FeelType::String), resolve("Tier"));
  // Neither a simple type nor an item definition of this model.
  assert_eq!(None, resolve("NoSuchType"));
  assert_eq!(None, resolve("date and time"));
  // An item definition belongs to its model's namespace.
  assert_eq!(None, evaluator.information_item_type("https://dsntk.io/elsewhere", "Tier"));
}
