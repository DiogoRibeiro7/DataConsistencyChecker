// Arithmatex wraps formulas so MathJax leaves code and ordinary prose alone.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex",
  },
};
