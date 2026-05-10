<script setup lang="ts">
import { ref } from 'vue'
import { useDiffusionStore } from '@/stores/diffusion'

const store = useDiffusionStore()

const snapshotPath = ref('')
const prompt = ref('an astronaut riding a horse on mars, photorealistic, 4k')
const negativePrompt = ref('')
const steps = ref(30)
const cfg = ref(7.5)
const seed = ref(42)
const size = ref(512)
</script>

<template>
  <div class="grid grid-cols-1 gap-4 xl:grid-cols-2">
    <section class="panel space-y-3">
      <h2 class="text-sm font-semibold">Load SD-1.5 snapshot</h2>
      <label class="block">
        <span class="label">Snapshot directory</span>
        <input
          v-model="snapshotPath"
          class="input"
          placeholder="/path/to/sd-1-5 (containing tokenizer/, text_encoder/, unet/, vae/)"
        />
      </label>
      <button
        class="btn btn-primary"
        :disabled="store.loading || !snapshotPath"
        @click="store.load(snapshotPath)"
      >
        {{ store.loading ? 'Loading…' : 'Load' }}
      </button>
      <div v-if="store.loaded" class="text-xs text-zinc-300">
        loaded · backend {{ store.loaded.backend }}
      </div>

      <hr class="border-zinc-800" />
      <h3 class="text-sm font-semibold">Generate</h3>
      <label class="block">
        <span class="label">Prompt</span>
        <textarea v-model="prompt" rows="3" class="input font-mono" />
      </label>
      <label class="block">
        <span class="label">Negative prompt</span>
        <textarea v-model="negativePrompt" rows="2" class="input font-mono" />
      </label>
      <div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <label class="block">
          <span class="label">Steps</span>
          <input v-model.number="steps" type="number" class="input" />
        </label>
        <label class="block">
          <span class="label">CFG</span>
          <input v-model.number="cfg" type="number" step="0.5" class="input" />
        </label>
        <label class="block">
          <span class="label">Seed</span>
          <input v-model.number="seed" type="number" class="input" />
        </label>
        <label class="block">
          <span class="label">Size</span>
          <input v-model.number="size" type="number" step="64" class="input" />
        </label>
      </div>
      <button
        class="btn btn-primary"
        :disabled="store.generating || !store.loaded"
        @click="
          store.generate({
            prompt,
            negativePrompt,
            steps,
            cfg,
            seed,
            size,
          })
        "
      >
        {{ store.generating ? 'Generating…' : 'Generate' }}
      </button>
    </section>

    <section class="panel">
      <h2 class="mb-2 text-sm font-semibold">Output</h2>
      <div
        v-if="!store.imageUrl"
        class="flex h-96 items-center justify-center rounded border border-dashed border-zinc-800 text-zinc-500"
      >
        nothing yet
      </div>
      <div v-else class="space-y-2">
        <img :src="store.imageUrl" class="w-full rounded" />
        <div class="text-xs text-zinc-400">{{ store.lastElapsedMs }} ms</div>
      </div>
    </section>

    <div v-if="store.error" class="col-span-full rounded border border-red-900 bg-red-950/40 p-2 text-sm text-red-300">
      {{ store.error }}
    </div>
  </div>
</template>
