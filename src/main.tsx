import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Intercept benign TFLite WASM / CPU delegate informational logs routed to stderr/console.error
const origConsoleError = console.error;
console.error = function (...args: any[]) {
  const first = args[0];
  const str = (typeof first === 'string' ? first : (first && first.message) || '') + '';
  if (
    str.includes('Created TensorFlow Lite XNNPACK delegate') ||
    str.includes('INFO: Created') ||
    str.includes('interrupted by a new load request') ||
    (first && first.name === 'AbortError')
  ) {
    console.info(...args);
    return;
  }
  origConsoleError.apply(console, args);
};

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
