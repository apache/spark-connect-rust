//! Server-gated end-to-end test of reattaching after a dropped connection. A TCP proxy in
//! front of the server resets the connection in the middle of a large result, and the
//! query still returns every row.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_reattach_drop`

use std::sync::{Arc, Mutex};

use spark_connect::SparkSession;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const ROWS: i64 = 3_000_000;
/// The response bytes that a connection carries before the proxy resets it.
const RESET_AFTER: usize = 2_000_000;

/// Takes one of the resets that are still due, if any.
fn take_reset(resets_due: &Mutex<usize>) -> bool {
    let mut due = resets_due.lock().unwrap();
    if *due == 0 {
        return false;
    }
    *due -= 1;
    true
}

/// Relays one client connection to the server. Once it has carried `RESET_AFTER` response
/// bytes, and a reset is still due, it aborts the client connection with a RST.
async fn relay(mut client: TcpStream, mut server: TcpStream, resets_due: Arc<Mutex<usize>>) {
    let mut from_client = vec![0; 64 * 1024];
    let mut from_server = vec![0; 64 * 1024];
    let mut received = 0;
    loop {
        tokio::select! {
            read = client.read(&mut from_client) => {
                let n = match read {
                    Ok(n) if n > 0 => n,
                    _ => return,
                };
                if server.write_all(&from_client[..n]).await.is_err() {
                    return;
                }
            }
            read = server.read(&mut from_server) => {
                let n = match read {
                    Ok(n) if n > 0 => n,
                    _ => return,
                };
                received += n;
                if client.write_all(&from_server[..n]).await.is_err() {
                    return;
                }
                if received >= RESET_AFTER && take_reset(&resets_due) {
                    // With SO_LINGER 0, closing the client socket sends a RST instead of a FIN.
                    client.set_zero_linger().unwrap();
                    return;
                }
            }
        }
    }
}

/// Starts the proxy in front of the server at `remote`, and returns the URL to connect to
/// and the number of resets that have not happened yet.
async fn start_proxy(remote: &str, resets: usize) -> (String, Arc<Mutex<usize>>) {
    let server = remote
        .trim_start_matches("sc://")
        .split(['/', ';'])
        .next()
        .unwrap()
        .to_string();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let resets_due = Arc::new(Mutex::new(resets));
    let due = resets_due.clone();
    tokio::spawn(async move {
        loop {
            let (client, _) = listener.accept().await.unwrap();
            let upstream = TcpStream::connect(&server).await.unwrap();
            tokio::spawn(relay(client, upstream, due.clone()));
        }
    });
    (format!("sc://{address}"), resets_due)
}

/// Collects `spark.range(ROWS)` through a proxy that resets the connection `resets` times,
/// and checks that every row arrives.
fn collect_through_resets(resets: usize) {
    let Ok(remote) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (url, resets_due) = runtime.block_on(start_proxy(&remote, resets));

    let spark = SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .unwrap();
    let rows = spark.range(ROWS).unwrap().collect().unwrap();

    assert_eq!(*resets_due.lock().unwrap(), 0, "the proxy did not reset");
    assert_eq!(rows.len(), ROWS as usize);
    let sum: i64 = rows
        .iter()
        .map(|row| row.get(0).unwrap().as_i64().unwrap())
        .sum();
    assert_eq!(sum, ROWS * (ROWS - 1) / 2);
}

#[test]
fn collect_reattaches_after_a_connection_reset() {
    collect_through_resets(1);
}

#[test]
fn collect_reattaches_after_three_connection_resets() {
    collect_through_resets(3);
}
