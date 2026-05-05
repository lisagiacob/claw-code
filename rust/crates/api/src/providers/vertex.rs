use std::collections::VecDeque;

use reqwest::StatusCode;
use serde_json::Value;
use telemetry::{AnthropicRequestProfile, ClientIdentity};

use crate::error::ApiError;
use crate::providers::{self, Provider, ProviderFuture};
use crate::sse::SseParser;
use crate::types::{MessageRequest, MessageResponse, StreamEvent};

pub const DEFAULT_VERTEX_LOCATION: &str = "global";
pub const DEFAULT_VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";
pub const GOOGLE_VERTEX_ANTHROPIC_PREFIX: &str = "google-vertex-anthropic/";
pub const VERTEX_ANTHROPIC_PREFIX: &str = "vertex-anthropic/";

#[derive(Debug, Clone)]
pub struct VertexAnthropicClient {
    http: reqwest::Client,
    project_id: String,
    location: String,
    access_token: String,
    request_profile: AnthropicRequestProfile,
}

impl VertexAnthropicClient {
    pub fn from_env() -> Result<Self, ApiError> {
        let project_id = read_required_env("ANTHROPIC_VERTEX_PROJECT_ID")?;
        let location = std::env::var("ANTHROPIC_VERTEX_LOCATION")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_VERTEX_LOCATION.to_string());
        let access_token = read_required_env("GOOGLE_ACCESS_TOKEN")?;
        Ok(Self::new(project_id, location, access_token))
    }

    #[must_use]
    pub fn new(
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Self {
        Self {
            http: crate::http_client::build_http_client_or_default(),
            project_id: project_id.into(),
            location: location.into(),
            access_token: access_token.into(),
            request_profile: AnthropicRequestProfile::new(ClientIdentity::default()),
        }
    }

    #[must_use]
    pub fn endpoint_for_model(&self, model: &str, streaming: bool) -> String {
        vertex_endpoint(
            &self.project_id,
            &self.location,
            &normalize_vertex_model_id(model),
            streaming,
        )
    }

    #[must_use]
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    #[must_use]
    pub fn location(&self) -> &str {
        &self.location
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let request = MessageRequest {
            stream: false,
            model: normalize_vertex_model_id(&request.model),
            ..request.clone()
        };
        providers::preflight_message_request(&request)?;
        let url = self.endpoint_for_model(&request.model, false);
        let body = self.vertex_body(&request)?;
        let response = self
            .build_request(&url)
            .json(&body)
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        let request_id = request_id_from_headers(response.headers());
        let text = response.text().await.map_err(ApiError::from)?;
        let mut parsed = serde_json::from_str::<MessageResponse>(&text)
            .map_err(|error| ApiError::json_deserialize("Vertex Anthropic", &request.model, &text, error))?;
        if parsed.request_id.is_none() {
            parsed.request_id = request_id;
        }
        Ok(parsed)
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        let request = MessageRequest {
            stream: true,
            model: normalize_vertex_model_id(&request.model),
            ..request.clone()
        };
        providers::preflight_message_request(&request)?;
        let url = self.endpoint_for_model(&request.model, true);
        let body = self.vertex_body(&request)?;
        let response = self
            .build_request(&url)
            .json(&body)
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        Ok(MessageStream {
            request_id: request_id_from_headers(response.headers()),
            parser: SseParser::new().with_context("Vertex Anthropic", request.model.clone()),
            response,
            pending: VecDeque::new(),
            done: false,
        })
    }

    fn build_request(&self, url: &str) -> reqwest::RequestBuilder {
        self.http
            .post(url)
            .header("content-type", "application/json")
            .bearer_auth(&self.access_token)
    }

    fn vertex_body(&self, request: &MessageRequest) -> Result<Value, ApiError> {
        vertex_body_with_profile(request, &self.request_profile).map_err(ApiError::from)
    }
}

impl Provider for VertexAnthropicClient {
    type Stream = MessageStream;

    fn send_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, MessageResponse> {
        Box::pin(async move { self.send_message(request).await })
    }

    fn stream_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, Self::Stream> {
        Box::pin(async move { self.stream_message(request).await })
    }
}

#[derive(Debug)]
pub struct MessageStream {
    request_id: Option<String>,
    response: reqwest::Response,
    parser: SseParser,
    pending: VecDeque<StreamEvent>,
    done: bool,
}

impl MessageStream {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }
            if self.done {
                let remaining = self.parser.finish()?;
                self.pending.extend(remaining);
                if let Some(event) = self.pending.pop_front() {
                    return Ok(Some(event));
                }
                return Ok(None);
            }
            match self.response.chunk().await? {
                Some(chunk) => self.pending.extend(self.parser.push(&chunk)?),
                None => self.done = true,
            }
        }
    }
}

#[must_use]
pub fn normalize_vertex_model_id(model: &str) -> String {
    let trimmed = model.trim();
    if let Some(model) = trimmed.strip_prefix(GOOGLE_VERTEX_ANTHROPIC_PREFIX) {
        return crate::providers::resolve_model_alias(model);
    }
    if let Some(model) = trimmed.strip_prefix(VERTEX_ANTHROPIC_PREFIX) {
        return crate::providers::resolve_model_alias(model);
    }
    crate::providers::resolve_model_alias(trimmed)
}

#[must_use]
pub fn is_vertex_model(model: &str) -> bool {
    let trimmed = model.trim();
    trimmed.starts_with(GOOGLE_VERTEX_ANTHROPIC_PREFIX)
        || trimmed.starts_with(VERTEX_ANTHROPIC_PREFIX)
}

#[must_use]
pub fn vertex_endpoint(project_id: &str, location: &str, model: &str, streaming: bool) -> String {
    let suffix = if streaming {
        "streamRawPredict"
    } else {
        "rawPredict"
    };
    format!(
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model}:{suffix}"
    )
}

pub fn vertex_body_with_profile(
    request: &MessageRequest,
    profile: &AnthropicRequestProfile,
) -> Result<Value, serde_json::Error> {
    let request = MessageRequest {
        model: normalize_vertex_model_id(&request.model),
        ..request.clone()
    };
    let mut body = profile.render_json_body(&request)?;
    let object = body.as_object_mut().ok_or_else(|| {
        serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "request body must serialize to a JSON object",
        ))
    })?;
    object.remove("model");
    object.insert(
        "anthropic_version".to_string(),
        Value::String(DEFAULT_VERTEX_ANTHROPIC_VERSION.to_string()),
    );
    Ok(body)
}

fn read_required_env(name: &'static str) -> Result<String, ApiError> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ApiError::missing_credentials("Vertex Anthropic", vertex_env_vars(name)))
}

fn vertex_env_vars(name: &'static str) -> &'static [&'static str] {
    match name {
        "ANTHROPIC_VERTEX_PROJECT_ID" => &["ANTHROPIC_VERTEX_PROJECT_ID"],
        "GOOGLE_ACCESS_TOKEN" => &["GOOGLE_ACCESS_TOKEN"],
        _ => &[
            "ANTHROPIC_VERTEX_PROJECT_ID",
            "ANTHROPIC_VERTEX_LOCATION",
            "GOOGLE_ACCESS_TOKEN",
        ],
    }
}

fn request_id_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get("request-id")
        .or_else(|| headers.get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let request_id = request_id_from_headers(response.headers());
    let body = response.text().await.unwrap_or_else(|_| String::new());
    let retryable = is_retryable_status(status);
    let parsed = serde_json::from_str::<VertexErrorEnvelope>(&body).ok();
    Err(ApiError::Api {
        status,
        error_type: parsed
            .as_ref()
            .and_then(|error| error.error.status.clone()),
        message: parsed.as_ref().map(|error| error.error.message.clone()),
        request_id,
        body,
        retryable,
        suggested_action: None,
    })
}

const fn is_retryable_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 429 | 500 | 502 | 503 | 504)
}

#[derive(Debug, serde::Deserialize)]
struct VertexErrorEnvelope {
    error: VertexError,
}

#[derive(Debug, serde::Deserialize)]
struct VertexError {
    #[serde(default)]
    message: String,
    #[serde(default)]
    status: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use telemetry::AnthropicRequestProfile;

    use super::{
        is_vertex_model, normalize_vertex_model_id, vertex_body_with_profile, vertex_endpoint,
        DEFAULT_VERTEX_ANTHROPIC_VERSION,
    };
    use crate::types::{InputMessage, MessageRequest};

    #[test]
    fn vertex_model_prefix_is_stripped_and_aliases_resolve() {
        assert!(is_vertex_model("google-vertex-anthropic/sonnet"));
        assert_eq!(
            normalize_vertex_model_id("google-vertex-anthropic/sonnet"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            normalize_vertex_model_id("vertex-anthropic/claude-opus-4-6"),
            "claude-opus-4-6"
        );
    }

    #[test]
    fn vertex_endpoint_uses_model_in_url() {
        assert_eq!(
            vertex_endpoint("p1", "global", "claude-sonnet-4-6", true),
            "https://global-aiplatform.googleapis.com/v1/projects/p1/locations/global/publishers/anthropic/models/claude-sonnet-4-6:streamRawPredict"
        );
        assert_eq!(
            vertex_endpoint("p1", "us-east5", "claude-sonnet-4-6", false),
            "https://us-east5-aiplatform.googleapis.com/v1/projects/p1/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-6:rawPredict"
        );
    }

    #[test]
    fn vertex_body_removes_model_and_adds_vertex_anthropic_version() {
        let request = MessageRequest {
            model: "google-vertex-anthropic/sonnet".to_string(),
            max_tokens: 100,
            messages: vec![InputMessage::user_text("hello")],
            ..MessageRequest::default()
        };
        let body = vertex_body_with_profile(&request, &AnthropicRequestProfile::default())
            .expect("body should render");
        assert_eq!(
            body.get("anthropic_version"),
            Some(&json!(DEFAULT_VERTEX_ANTHROPIC_VERSION))
        );
        assert!(body.get("model").is_none());
        assert_eq!(body.get("max_tokens"), Some(&json!(100)));
    }
}
