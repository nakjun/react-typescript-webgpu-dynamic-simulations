import React, { useEffect } from 'react';
import { Initialize } from './test';

function App() {
  useEffect(() => {
    Initialize();
  }, []);

  return (
    <div className="App">
      <canvas id="gfx-main" width="800" height="600"></canvas>
    </div>
  );
}

export default App;
