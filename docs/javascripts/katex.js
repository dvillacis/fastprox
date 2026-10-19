// Renders the math that pymdownx.arithmatex marks (generic mode: \( \) and
// \[ \]) with KaTeX, on every page load, including Material's instant loads.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
});
