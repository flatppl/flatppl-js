(function () {
  "use strict";

  var root = document.documentElement;
  var storageKey = "flatppl-theme";
  var systemTheme = matchMedia("(prefers-color-scheme: dark)");
  var modes = ["system", "light", "dark"];
  var labels = { system: "System", light: "Whiteboard", dark: "Blackboard" };
  var sharedDomain = location.hostname === "flatppl.org" || location.hostname.endsWith(".flatppl.org");
  root.dataset.fpShellJs = "";

  function storedTheme() {
    try {
      if (sharedDomain) {
        var cookie = document.cookie.split("; ").find(function (part) {
          return part.indexOf(storageKey + "=") === 0;
        });
        var shared = cookie && cookie.slice(storageKey.length + 1);
        if (modes.indexOf(shared) !== -1) return shared;
      }
      var value = localStorage.getItem(storageKey);
      return value === "light" || value === "dark" ? value : null;
    } catch (_) {
      return null;
    }
  }

  var mode = storedTheme() || "system";

  function setTheme(nextMode, persist) {
    mode = nextMode;
    root.dataset.theme = mode === "system" ? (systemTheme.matches ? "dark" : "light") : mode;
    if (persist) {
      try {
        if (sharedDomain) {
          document.cookie = storageKey + "=" + mode + "; Domain=flatppl.org; Path=/; Max-Age=31536000; SameSite=Lax; Secure";
        }
        localStorage.setItem(storageKey, mode);
      } catch (_) {
        // The theme still applies when storage is unavailable.
      }
    }
    document.querySelectorAll("[data-fp-theme-toggle]").forEach(function (button) {
      var target = labels[modes[(modes.indexOf(mode) + 1) % modes.length]];
      var label = button.querySelector("[data-fp-theme-label]");
      if (label) label.textContent = labels[mode];
      button.setAttribute("aria-label", "Theme: " + labels[mode] + ". Switch to " + target);
    });
  }

  function bindShell() {
    setTheme(mode, false);

    document.querySelectorAll("[data-fp-theme-toggle]").forEach(function (button) {
      button.addEventListener("click", function () {
        setTheme(modes[(modes.indexOf(mode) + 1) % modes.length], true);
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

  setTheme(mode, false);
  systemTheme.addEventListener("change", function () {
    if (mode === "system") setTheme(mode, false);
  });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bindShell);
  else bindShell();
})();
