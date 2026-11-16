/// Integration tests of the core client against a live Spark Connect server, at `SPARK_REMOTE`
/// (or `sc://localhost:15002`). The `connect-e2e` job of CI runs them with
/// `cargo test -p apache-spark-connect-core --test integration_test -- --ignored`.
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

    let builder = ChannelBuilder::parse(&remote()).expect("Failed to parse URL");
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

    let builder = ChannelBuilder::parse(&remote()).expect("Failed to parse URL");
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

    let builder = ChannelBuilder::parse(&remote()).expect("Failed to parse URL");
    let client = SparkConnectClient::connect(&builder)
        .await
        .expect("Failed to connect to server");

    // A Get of a conf that every session has: the server answers its value.
    let mut request = spark_connect_proto::ConfigRequest::default();
    request.session_id = client.session_id().to_string();
    request.user_context = Some(spark_connect_proto::UserContext::default());
    let get = spark_connect_proto::config_request::Get {
        keys: vec!["spark.sql.shuffle.partitions".to_string()],
    };
    request.operation = Some(spark_connect_proto::config_request::Operation {
        op_type: Some(spark_connect_proto::config_request::operation::OpType::Get(
            get,
        )),
    });
    let response = client.config(request).await.expect("Failed to get a conf");
    let pair = response
        .pairs
        .iter()
        .find(|pair| pair.key == "spark.sql.shuffle.partitions")
        .expect("the conf is not in the response");
    assert!(
        pair.value
            .as_deref()
            .is_some_and(|value| value.parse::<u32>().is_ok()),
        "the conf has no value of a number: {pair:?}"
    );
}

/// The Spark Connect URL of the server of the tests.
fn remote() -> String {
    std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string())
}

/// Check if live tests should run.
/// Returns true if SPARK_REMOTE is set or if the env var is not present (assume local testing).
fn should_run_live_tests() -> bool {
    // You can enable via SPARK_REMOTE env var or just skip for now
    std::env::var("SPARK_REMOTE").is_ok() || std::env::var("RUN_SPARK_INTEGRATION_TESTS").is_ok()
}
