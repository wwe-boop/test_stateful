class PcmStreamPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.current = null;
    this.offset = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type === "push" && event.data.samples) {
        this.queue.push(new Float32Array(event.data.samples));
      }
      if (event.data?.type === "clear") {
        this.queue = [];
        this.current = null;
        this.offset = 0;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    for (let i = 0; i < output.length; i += 1) {
      if (!this.current || this.offset >= this.current.length) {
        this.current = this.queue.shift() || null;
        this.offset = 0;
      }
      output[i] = this.current ? this.current[this.offset++] : 0;
    }
    return true;
  }
}

registerProcessor("pcm-stream-player", PcmStreamPlayerProcessor);

