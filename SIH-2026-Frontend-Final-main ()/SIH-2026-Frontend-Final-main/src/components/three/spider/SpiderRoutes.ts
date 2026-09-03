export type SpiderDepth = "foreground" | "behind-panel";

export type SpiderRouteBeat = {
  id: string;
  duration: number;
  depth: SpiderDepth;
  action: "crawl" | "anchor-web" | "rappel" | "pause" | "exit";
  /** Viewport-relative coordinates. Renderer converts these to its camera space. */
  from: readonly [number, number];
  to: readonly [number, number];
};

/**
 * The first authored journey. It deliberately does not loop: the route is
 * replayed only after a future, explicit route scheduler is added.
 */
export const homeSpiderRoute: readonly SpiderRouteBeat[] = [
  { id: "enter-right", duration: 2400, depth: "foreground", action: "crawl", from: [1.08, 0.1], to: [0.9, 0.22] },
  { id: "edge-crawl", duration: 2600, depth: "foreground", action: "crawl", from: [0.9, 0.22], to: [0.76, 0.42] },
  { id: "anchor", duration: 700, depth: "foreground", action: "anchor-web", from: [0.76, 0.42], to: [0.76, 0.42] },
  { id: "rappel", duration: 2100, depth: "foreground", action: "rappel", from: [0.76, 0.42], to: [0.66, 0.57] },
  { id: "pipeline-behind", duration: 2800, depth: "behind-panel", action: "crawl", from: [0.66, 0.57], to: [0.47, 0.67] },
  { id: "reemerge", duration: 1700, depth: "foreground", action: "crawl", from: [0.47, 0.67], to: [0.31, 0.73] },
  { id: "exit", duration: 1900, depth: "foreground", action: "exit", from: [0.31, 0.73], to: [-0.12, 0.84] },
];
