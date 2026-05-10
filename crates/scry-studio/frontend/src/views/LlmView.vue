<script setup lang="ts">
import { ref } from 'vue'
import { useLlmStore } from '@/stores/llm'

const store = useLlmStore()

const modelPath = ref('')
const tokenizerPath = ref('')
const preset = ref<'gpt2'>('gpt2')

const prompt = ref('Once upon a time')
const maxTokens = ref(64)
const temperature = ref(1.0)
const topK = ref(40)
const topP = ref(0.95)
const seed = ref(42)
const inspectTopK = ref(20)

const tokenizeText = ref('hello world')
</script>

<template>
  <div class="grid grid-cols-1 gap-4 xl:grid-cols-2">
    <section class="panel space-y-3">
      <h2 class="text-sm font-semibold">Load model</h2>
      <div class="space-y-2">
        <label class="block">
          <span class="label">Safetensors path</span>
          <input v-model="modelPath" class="input" placeholder="/path/to/gpt2/model.safetensors" />
        </label>
        <label class="block">
          <span class="label">Tokenizer (tokenizer.json)</span>
          <input v-model="tokenizerPath" class="input" placeholder="/path/to/tokenizer.json" />
        </label>
        <label class="block">
          <span class="label">Preset</span>
          <select v-model="preset" class="input">
            <option value="gpt2">gpt2 (small, 124M)</option>
          </select>
        </label>
        <button
          class="btn btn-primary"
          :disabled="store.loading || !modelPath || !tokenizerPath"
          @click="store.load({ modelPath, tokenizerPath, preset })"
        >
          {{ store.loading ? 'Loading…' : 'Load' }}
        </button>
      </div>
      <div v-if="store.loaded" class="rounded bg-zinc-950 p-2 text-xs text-zinc-300">
        loaded · vocab {{ store.loaded.vocabSize }} · ctx {{ store.loaded.maxSeqLen }} ·
        layers {{ store.loaded.nLayers }} · d {{ store.loaded.dModel }}
      </div>
    </section>

    <section class="panel space-y-3">
      <h2 class="text-sm font-semibold">Generate</h2>
      <textarea v-model="prompt" rows="4" class="input font-mono" />
      <div class="grid grid-cols-2 gap-2 sm:grid-cols-5">
        <label class="block">
          <span class="label">max</span>
          <input v-model.number="maxTokens" type="number" class="input" />
        </label>
        <label class="block">
          <span class="label">temp</span>
          <input v-model.number="temperature" type="number" step="0.05" class="input" />
        </label>
        <label class="block">
          <span class="label">top-k</span>
          <input v-model.number="topK" type="number" class="input" />
        </label>
        <label class="block">
          <span class="label">top-p</span>
          <input v-model.number="topP" type="number" step="0.05" class="input" />
        </label>
        <label class="block">
          <span class="label">seed</span>
          <input v-model.number="seed" type="number" class="input" />
        </label>
      </div>
      <button
        class="btn btn-primary"
        :disabled="store.generating || !store.loaded"
        @click="
          store.generate({
            prompt,
            maxTokens,
            temperature,
            topK,
            topP,
            seed,
          })
        "
      >
        {{ store.generating ? 'Generating…' : 'Generate' }}
      </button>
      <div v-if="store.lastOutput" class="space-y-1">
        <div class="label">Output · {{ store.lastElapsedMs }} ms</div>
        <pre class="whitespace-pre-wrap rounded bg-zinc-950 p-3 font-mono text-sm">{{ store.lastOutput }}</pre>
      </div>
    </section>

    <section class="panel space-y-3">
      <h2 class="text-sm font-semibold">Tokenizer</h2>
      <textarea v-model="tokenizeText" rows="2" class="input font-mono" />
      <div class="flex gap-2">
        <button class="btn" :disabled="!store.loaded" @click="store.tokenize(tokenizeText)">
          Encode
        </button>
        <button
          v-if="store.tokenization"
          class="btn"
          @click="store.decode(store.tokenization.ids)"
        >
          Decode →
        </button>
      </div>
      <div v-if="store.tokenization" class="rounded bg-zinc-950 p-2 text-xs">
        <div class="label mb-1">{{ store.tokenization.ids.length }} tokens</div>
        <div class="flex flex-wrap gap-1 font-mono">
          <span
            v-for="(id, i) in store.tokenization.ids"
            :key="i"
            class="rounded bg-zinc-800 px-1.5 py-0.5"
          >
            <span class="text-zinc-400">{{ id }}</span>
            <span class="ml-1 text-zinc-200">{{ store.tokenization.pieces[i] }}</span>
          </span>
        </div>
      </div>
    </section>

    <section class="panel space-y-3">
      <h2 class="text-sm font-semibold">Next-token logits</h2>
      <div class="flex gap-2">
        <input v-model.number="inspectTopK" type="number" class="input w-24" />
        <button
          class="btn"
          :disabled="!store.loaded"
          @click="store.inspectLogits(prompt, inspectTopK)"
        >
          Inspect
        </button>
      </div>
      <table v-if="store.logits" class="w-full text-xs font-mono">
        <thead class="text-zinc-400">
          <tr>
            <th class="text-left">id</th>
            <th class="text-left">piece</th>
            <th class="text-right">logit</th>
            <th class="text-right">prob</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="t in store.logits.top" :key="t.id" class="border-t border-zinc-800">
            <td>{{ t.id }}</td>
            <td class="truncate">{{ t.piece }}</td>
            <td class="text-right">{{ t.logit.toFixed(3) }}</td>
            <td class="text-right">{{ (t.prob * 100).toFixed(2) }}%</td>
          </tr>
        </tbody>
      </table>
    </section>

    <div v-if="store.error" class="col-span-full rounded border border-red-900 bg-red-950/40 p-2 text-sm text-red-300">
      {{ store.error }}
    </div>
  </div>
</template>
