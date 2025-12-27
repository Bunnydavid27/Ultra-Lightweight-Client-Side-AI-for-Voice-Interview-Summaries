let session;
let tokenizer;

async function load() {
  session = await ort.InferenceSession.create(
    "t5_summarizer_quant.onnx",
    { executionProviders: ["wasm"] }
  );

  tokenizer = await window.transformers.AutoTokenizer.from_pretrained(
    "t5-small"
  );

  console.log("Model + tokenizer loaded");
}

load();

function argmax(arr) {
  return arr.indexOf(Math.max(...arr));
}

async function run() {
  const text = document.getElementById("input").value;

  const enc = await tokenizer(
    "summarize: " + text,
    { return_tensors: "np" }
  );

  let input_ids = enc.input_ids;
  let attention_mask = enc.attention_mask;

  let decoder_ids = [[tokenizer.token_to_id("<pad>")]];
  let output_ids = [];

  for (let step = 0; step < 40; step++) {
    const feeds = {
      input_ids: new ort.Tensor("int64", input_ids.data, input_ids.shape),
      attention_mask: new ort.Tensor("int64", attention_mask.data, attention_mask.shape),
      decoder_input_ids: new ort.Tensor("int64", BigInt64Array.from(decoder_ids.flat()), [1, decoder_ids[0].length])
    };

    const results = await session.run(feeds);
    const logits = results.logits.data;

    const vocab = tokenizer.vocab_size;
    const lastTokenLogits = logits.slice(
      (decoder_ids[0].length - 1) * vocab,
      decoder_ids[0].length * vocab
    );

    const nextToken = argmax(Array.from(lastTokenLogits));
    output_ids.push(nextToken);

    if (nextToken === tokenizer.token_to_id("</s>")) break;

    decoder_ids[0].push(nextToken);
  }

  const summary = tokenizer.decode(output_ids, { skip_special_tokens: true });
  document.getElementById("output").innerText = summary;
}
