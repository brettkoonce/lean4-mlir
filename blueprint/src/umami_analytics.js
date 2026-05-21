// Umami Analytics — privacy-friendly, cookieless pageview tracking.
// Loaded on every blueprint page via plastex.cfg's `extra-js`.
//
// The standard Umami snippet uses a custom <script data-website-id="...">
// attribute that plasTeX's extra-js mechanism can't emit directly, so we
// build the tag dynamically here. Same end result.
(function () {
  var s = document.createElement('script');
  s.defer = true;
  s.src = 'https://cloud.umami.is/script.js';
  s.setAttribute('data-website-id', 'df07a183-ef82-4758-8b6f-1d888ba3ec5b');
  document.head.appendChild(s);
})();
