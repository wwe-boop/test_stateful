export function wavBlobFromPcmF32(chunks: ArrayBuffer[], sampleRate: number): Blob {
  const byteLength = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const pcm = new Float32Array(byteLength / 4);
  let offset = 0;
  for (const chunk of chunks) {
    const part = new Float32Array(chunk);
    pcm.set(part, offset);
    offset += part.length;
  }

  const dataSize = pcm.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let cursor = 44;
  for (const sample of pcm) {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(cursor, Math.round(clipped * 32767), true);
    cursor += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}
