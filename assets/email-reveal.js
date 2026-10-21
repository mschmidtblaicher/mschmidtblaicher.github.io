// Lightweight email obfuscation: the address is only assembled after the
// visitor asks for it. This deters naive scrapers; it is not a security boundary.
(function () {
  "use strict";

  var PARTS = { user: ["mschmidt", "blaicher"], domain: ["gmail", "com"] };
  var DURATION_MS = 700;
  var SCRAMBLE_INTERVAL_MS = 35;
  var GLYPHS = "abcdefghijklmnopqrstuvwxyz0123456789@._-#%&";

  function assembleAddress() {
    return PARTS.user.join("") + String.fromCharCode(64) + PARTS.domain.join(".");
  }

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function randomGlyph() {
    return GLYPHS.charAt(Math.floor(Math.random() * GLYPHS.length));
  }

  // Characters settle left to right; unsettled ones keep cycling.
  function scramble(target, progress) {
    var settled = Math.floor(progress * target.length);
    var out = target.slice(0, settled);
    for (var i = settled; i < target.length; i++) out += randomGlyph();
    return out;
  }

  function animate(el, target, done) {
    var start = null;
    var lastUpdate = -Infinity;
    function frame(now) {
      if (start === null) start = now;
      var progress = Math.min((now - start) / DURATION_MS, 1);
      if (progress === 1) {
        el.textContent = target;
        done();
        return;
      }
      if (now - lastUpdate >= SCRAMBLE_INTERVAL_MS) {
        el.textContent = scramble(target, progress);
        lastUpdate = now;
      }
      window.requestAnimationFrame(frame);
    }
    window.requestAnimationFrame(frame);
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try {
        ok = document.execCommand("copy");
      } catch (e) {
        ok = false;
      }
      document.body.removeChild(area);
      ok ? resolve() : reject(new Error("copy failed"));
    });
  }

  function init(root) {
    var trigger = root.querySelector(".email-reveal__trigger");
    var address = root.querySelector(".email-reveal__address");
    var status = root.querySelector(".email-reveal__status");
    var actions = root.querySelector(".email-reveal__actions");
    var copyButton = root.querySelector(".email-reveal__copy");
    var openLink = root.querySelector(".email-reveal__open");
    var revealing = false;
    var resetTimer = null;

    root.hidden = false;

    function finish(email) {
      address.classList.remove("is-scrambling");
      address.removeAttribute("aria-hidden");
      openLink.href = "mailto:" + email;
      actions.hidden = false;
      status.textContent = "Email address: " + email;
      trigger.hidden = true;
      // The trigger is gone, so keep keyboard focus in a sensible place.
      copyButton.focus();
    }

    trigger.addEventListener("click", function () {
      if (revealing) return;
      revealing = true;
      trigger.setAttribute("aria-disabled", "true");
      trigger.setAttribute("aria-expanded", "true");
      var email = assembleAddress();
      address.hidden = false;
      if (prefersReducedMotion()) {
        address.textContent = email;
        finish(email);
      } else {
        // Keep screen readers from reading out random characters.
        address.setAttribute("aria-hidden", "true");
        address.classList.add("is-scrambling");
        status.textContent = "Revealing email address";
        animate(address, email, function () {
          finish(email);
        });
      }
    });

    copyButton.addEventListener("click", function () {
      var email = address.textContent;
      copyText(email).then(
        function () {
          copyButton.textContent = "Copied";
          status.textContent = "Email address copied to clipboard";
        },
        function () {
          copyButton.textContent = "Copy failed";
          status.textContent = "Could not copy. The address is " + email;
        }
      );
      window.clearTimeout(resetTimer);
      resetTimer = window.setTimeout(function () {
        copyButton.textContent = "Copy";
      }, 2000);
    });
  }

  function start() {
    var roots = document.querySelectorAll(".email-reveal");
    for (var i = 0; i < roots.length; i++) init(roots[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
