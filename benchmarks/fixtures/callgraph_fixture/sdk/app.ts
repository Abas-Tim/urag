import * as core from "sdk/core";
import * as net from "sdk/net";

export function run(): void {
  core.makeClient();
}

export function ping(url: string): void {
  net.connect(url);
}
