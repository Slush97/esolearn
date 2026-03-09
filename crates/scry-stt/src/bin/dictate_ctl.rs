//! scry-dictate-ctl — sends commands to the scry-dictate daemon.
//!
//! Usage:
//!   scry-dictate-ctl start    # begin recording
//!   scry-dictate-ctl stop     # stop recording + transcribe
//!   scry-dictate-ctl toggle   # toggle recording on/off

use std::io::Write;
use std::os::unix::net::UnixStream;

const SOCKET_PATH: &str = "/tmp/scry-dictate.sock";

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    if cmd.is_empty() {
        eprintln!("usage: scry-dictate-ctl <start|stop>");
        std::process::exit(1);
    }

    let mut stream = match UnixStream::connect(SOCKET_PATH) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("scry-dictate-ctl: cannot connect to daemon: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = writeln!(stream, "{cmd}") {
        eprintln!("scry-dictate-ctl: write error: {e}");
        std::process::exit(1);
    }
}
