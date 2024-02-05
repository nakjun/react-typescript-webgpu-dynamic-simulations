import React, { useEffect, useRef } from 'react';
import { Initialize } from './RenderModule';

function App() {  
  
  let start = false;

  useEffect(() => {
    if(!start) {
      Initialize();
      start = true;
    }
  }, []);

  return (
    <div className="App">      
      <div id="fpsDisplay"></div>
      <canvas id="gfx-main" width="800" height="600"></canvas>
    </div>
  );
}

export default App;
