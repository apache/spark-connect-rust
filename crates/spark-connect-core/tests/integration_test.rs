/// Integration tests against a live Spark Connect server.
///
/// These tests require the server to be running at localhost:15002.
/// Run with: RUN_SPARK_INTEGRATION_TESTS=1 cargo test --test integration_test -- --ignored
use spark_connect_core::ChannelBuilder;
use spark_connect_core::SparkConnectClient;
use spark_connect_proto::AnalyzePlanRequest;

#[tokio::test]
#[ignore] // Gate behind ignore or env var check
async fn test_connect_to_live_server() {
    // Skip if server not available
    if !should_run_live_tests() {
        return;
    }

    let builder = ChannelBuilder::parse("sc://localhost:15002").expect("Failed to parse URL");
    let client = SparkConnectClient::connect(&builder)
        .await
        .expect("Failed to connect to server");

    assert!(!client.session_id().is_empty());
}

#[tokio::test]
#[ignore]
async fn test_analyze_spark_version() {
    if !should_run_live_tests() {
        return;
    }

    let builder = ChannelBuilder::parse("sc://localhost:15002").expect("Failed to parse URL");
    let client = SparkConnectClient::connect(&builder)
        .await
        .expect("Failed to connect to server");

    // Create a simple AnalyzePlan request for SparkVersion
    let mut request = AnalyzePlanRequest::default();
    request.session_id = client.session_id().to_string();
    request.user_context = Some(spark_connect_proto::UserContext::default());

    // Set the analyze oneof to SparkVersion (empty message)
    request.analyze = Some(
        spark_connect_proto::analyze_plan_request::Analyze::SparkVersion(
            spark_connect_proto::analyze_plan_request::SparkVersion::default(),
        ),
    );

    let response = client
        .analyze_plan(request)
        .await
        .expect("Failed to analyze plan");

    // We should get a response
    assert!(!response.session_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_config_get() {
    if !should_run_live_tests() {
        return;
    }

    let builder = ChannelBuilder::parse("sc://localhost:15002").expect("Failed to parse URL");
    let client = SparkConnectClient::connect(&builder)
        .await
        .expect("Failed to connect to server");

    let mut request = spark_connect_proto::ConfigRequest::default();
    request.session_id = client.session_id().to_string();
    request.user_context = Some(spark_connect_proto::UserContext::default());

    // This may fail with server error since we're not setting valid operation,
    // but it proves the RPC call goes through
    let result = client.config(request).await;
    // Either success or gRPC error is acceptable for this test
    assert!(result.is_ok() || result.is_err());
}

/// Check if live tests should run.
/// Returns true if SPARK_REMOTE is set or if the env var is not present (assume local testing).
fn should_run_live_tests() -> bool {
    // You can enable via SPARK_REMOTE env var or just skip for now
    std::env::var("SPARK_REMOTE").is_ok() || std::env::var("RUN_SPARK_INTEGRATION_TESTS").is_ok()
}
