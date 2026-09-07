(function () {
  "use strict";

  var root = document.documentElement;
  var storageKey = "flatppl-theme";
  root.dataset.fpShellJs = "";

  function storedTheme() {
    try {
      var value = localStorage.getItem(storageKey);
      return value === "light" || value === "dark" ? value : null;
    } catch (_) {
      return null;
    }
  }

  function preferredTheme() {
    var stored = storedTheme();
    if (stored) return stored;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function setTheme(theme, persist) {
    root.dataset.theme = theme;
    if (persist) {
      try {
        localStorage.setItem(storageKey, theme);
      } catch (_) {
        // The theme still applies when storage is unavailable.
      }
    }
    document.querySelectorAll("[data-fp-theme-toggle]").forEach(function (button) {
      var target = theme === "dark" ? "Whiteboard" : "Blackboard";
      var label = button.querySelector("[data-fp-theme-label]");
      if (label) label.textContent = target;
      button.setAttribute("aria-label", "Switch to " + target);
    });
  }

  function bindShell() {
    setTheme(root.dataset.theme || preferredTheme(), false);

    document.querySelectorAll("[data-fp-theme-toggle]").forEach(function (button) {
      button.addEventListener("click", function () {
        setTheme(root.dataset.theme === "dark" ? "light" : "dark", true);
      });
    });

    var active = root.dataset.fpActive;
    document.querySelectorAll("[data-fp-nav]").forEach(function (link) {
      if (link.dataset.fpNav === active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });

    document.querySelectorAll("[data-fp-menu-toggle]").forEach(function (button) {
      var menu = document.getElementById(button.getAttribute("aria-controls"));
      if (!menu) return;
      button.addEventListener("click", function () {
        var open = button.getAttribute("aria-expanded") !== "true";
        button.setAttribute("aria-expanded", String(open));
        if (open) menu.setAttribute("data-open", "");
        else menu.removeAttribute("data-open");
      });
    });
  }

  root.dataset.theme = preferredTheme();
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bindShell);
  else bindShell();
})();
