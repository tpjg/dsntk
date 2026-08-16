//! # Evaluator for Java external functions
//!
//! The transport is behind the `java-bridge` feature (off by default). With it
//! off there is no HTTP client in the build at all, and an `external`
//! invocation answers a FEEL `null` carrying a reason — the same shape this
//! module already returns for a malformed signature or an unreachable JVM, so
//! nothing above it changes.

use dsntk_feel::dto::ValueDto;
use dsntk_feel::value_null;
use dsntk_feel::values::Value;
#[cfg(feature = "java-bridge")]
use serde::Deserialize;
use serde::Serialize;
#[cfg(feature = "java-bridge")]
use std::sync::LazyLock;

#[cfg(feature = "java-bridge")]
static CLIENT: LazyLock<reqwest::blocking::Client> = LazyLock::new(reqwest::blocking::Client::new);

#[cfg(feature = "java-bridge")]
const JAVA_RPC_SERVER_URL: &str = "http://127.0.0.1:22023/api/rest/v1/rpc/evaluate";

#[derive(Serialize)]
struct RequestDto {
  /// Name of the Java class, where called method is defined.
  #[serde(rename = "className")]
  class_name: String,
  /// Name of the static method to be called.
  #[serde(rename = "methodName")]
  method_name: String,
  /// List of parameter types of the called method.
  #[serde(rename = "parameterTypes")]
  parameter_types: Vec<String>,
  /// Arguments to be passed to called method.
  #[serde(rename = "arguments")]
  argument_values: Vec<ValueDto>,
}

#[cfg(feature = "java-bridge")]
#[derive(Deserialize)]
struct ResponseDto {
  /// Response payload when calling external function succeeds.
  #[serde(rename = "data")]
  data: Option<ValueDto>,
  /// Error message on failure.
  #[serde(rename = "error")]
  error: Option<String>,
}

/// Evaluates external Java function.
pub fn evaluate_external_java_function(class_name: &str, method_signature: &str, arguments: &[Value]) -> Value {
  let mut parts = method_signature.trim().split('(');
  let Some(method_name) = parts.next() else {
    return value_null!("no method name in method signature");
  };
  let Some(parameter_type_names) = parts.next() else {
    return value_null!("no parameter types in method signature");
  };
  let parameter_types: Vec<String> = parameter_type_names
    .trim()
    .trim_end_matches(')')
    .split(',')
    .filter_map(|s| if s.trim().is_empty() { None } else { Some(s.trim().to_string()) })
    .collect();
  if parameter_types.len() != arguments.len() {
    return value_null!(
      "the number of parameter types ({}) differs from the number of arguments ({})",
      parameter_types.len(),
      arguments.len()
    );
  }
  let mut argument_values = vec![];
  for argument in arguments {
    match ValueDto::try_from(argument) {
      Ok(value_dto) => argument_values.push(value_dto),
      Err(reason) => return value_null!("{}", reason.to_string()),
    };
  }
  let request_dto = RequestDto {
    class_name: class_name.to_string(),
    method_name: method_name.to_string(),
    parameter_types,
    argument_values,
  };
  invoke(&request_dto)
}

/// Sends the prepared request to the Java RPC server.
#[cfg(feature = "java-bridge")]
fn invoke(request_dto: &RequestDto) -> Value {
  match CLIENT.post(JAVA_RPC_SERVER_URL).json(request_dto).send() {
    Ok(response) => match response.json::<ResponseDto>() {
      Ok(response_dto) => {
        if let Some(reason) = response_dto.error {
          value_null!("{}", reason)
        } else if let Some(value_dto) = response_dto.data {
          match Value::try_from(&value_dto) {
            Ok(value) => value,
            Err(reason) => value_null!("{}", reason),
          }
        } else {
          value_null!("no data in response")
        }
      }
      Err(reason) => value_null!("{}", reason),
    },
    Err(reason) => value_null!("{}", reason),
  }
}

/// Refuses the invocation: this build has no HTTP client.
///
/// A `null` with a reason rather than a panic or an error type.
#[cfg(not(feature = "java-bridge"))]
fn invoke(request_dto: &RequestDto) -> Value {
  value_null!(
    "external Java function '{}.{}' was not invoked: this build of dsntk-feel-evaluator was compiled without the 'java-bridge' feature, so it contains no HTTP client",
    request_dto.class_name,
    request_dto.method_name
  )
}
