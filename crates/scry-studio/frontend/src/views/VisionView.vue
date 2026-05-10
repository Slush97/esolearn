<script setup lang="ts">
import { computed, ref } from 'vue'
import { useVisionStore } from '@/stores/vision'

const store = useVisionStore()

const resnetPath = ref('')
const resnetPreset = ref<'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152'>('resnet50')
const labelsPath = ref('')

const scrfdPath = ref('')
const scrfdInputSize = ref(640)
const conf = ref(0.5)

const resnetFile = ref<File | null>(null)
const scrfdFile = ref<File | null>(null)

function onResnetFile(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0] ?? null
  resnetFile.value = f
}

function onScrfdFile(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0] ?? null
  scrfdFile.value = f
}

const detectionsOverlay = computed(() => store.detections?.detections ?? [])
</script>

<template>
  <div class="grid grid-cols-1 gap-4 xl:grid-cols-2">
    <section class="panel space-y-3">
      <h2 class="text-sm font-semibold">ResNet — image classification</h2>
      <label class="block">
        <span class="label">Safetensors path</span>
        <input v-model="resnetPath" class="input" placeholder="/path/to/resnet50.safetensors" />
      </label>
      <label class="block">
        <span class="label">Preset</span>
        <select v-model="resnetPreset" class="input">
          <option>resnet18</option>
          <option>resnet34</option>
          <option>resnet50</option>
          <option>resnet101</option>
          <option>resnet152</option>
        </select>
      </label>
      <label class="block">
        <span class="label">Labels file (optional, one class per line)</span>
        <input v-model="labelsPath" class="input" placeholder="/path/to/imagenet_labels.txt" />
      </label>
      <button
        class="btn btn-primary"
        :disabled="store.busy || !resnetPath"
        @click="store.loadResnet({ modelPath: resnetPath, preset: resnetPreset, labelsPath: labelsPath || undefined })"
      >
        Load ResNet
      </button>
      <div v-if="store.resnet" class="text-xs text-zinc-300">
        loaded · {{ store.resnet.numClasses }} classes · labels {{ store.resnet.hasLabels }}
      </div>

      <hr class="border-zinc-800" />
      <input type="file" accept="image/*" class="text-xs" @change="onResnetFile" />
      <button
        class="btn"
        :disabled="store.busy || !resnetFile || !store.resnet"
        @click="resnetFile && store.classify(resnetFile)"
      >
        Classify
      </button>

      <div v-if="store.classification" class="space-y-1">
        <div class="label">Top-{{ store.classification.top.length }} · {{ store.classification.elapsedMs }} ms</div>
        <table class="w-full text-xs">
          <tbody>
            <tr v-for="t in store.classification.top" :key="t.classId" class="border-t border-zinc-800">
              <td class="py-1 text-zinc-400">{{ t.classId }}</td>
              <td class="py-1">{{ t.label ?? '—' }}</td>
              <td class="py-1 text-right font-mono">{{ (t.score * 100).toFixed(2) }}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel space-y-3">
      <h2 class="text-sm font-semibold">SCRFD — face detection</h2>
      <label class="block">
        <span class="label">ONNX path</span>
        <input v-model="scrfdPath" class="input" placeholder="/path/to/scrfd_2.5g.onnx" />
      </label>
      <label class="block">
        <span class="label">Input size</span>
        <input v-model.number="scrfdInputSize" type="number" class="input" />
      </label>
      <button
        class="btn btn-primary"
        :disabled="store.busy || !scrfdPath"
        @click="store.loadScrfd(scrfdPath, scrfdInputSize)"
      >
        Load SCRFD
      </button>
      <div v-if="store.scrfd" class="text-xs text-zinc-300">
        loaded · input {{ store.scrfd.inputSize }}
      </div>

      <hr class="border-zinc-800" />
      <input type="file" accept="image/*" class="text-xs" @change="onScrfdFile" />
      <label class="block">
        <span class="label">Confidence threshold</span>
        <input v-model.number="conf" type="number" step="0.05" min="0" max="1" class="input" />
      </label>
      <button
        class="btn"
        :disabled="store.busy || !scrfdFile || !store.scrfd"
        @click="scrfdFile && store.detect(scrfdFile, conf)"
      >
        Detect
      </button>

      <div v-if="store.lastImage" class="relative inline-block">
        <img :src="store.lastImage" class="max-h-96 rounded" />
        <svg
          v-if="detectionsOverlay.length"
          class="pointer-events-none absolute inset-0 h-full w-full"
          preserveAspectRatio="none"
          :viewBox="`0 0 1000 1000`"
        >
          <!-- bbox coords are pixel-space; SVG just needs container scaling. -->
        </svg>
      </div>
      <div v-if="store.detections" class="text-xs text-zinc-300">
        {{ store.detections.detections.length }} faces · {{ store.detections.elapsedMs }} ms
      </div>
    </section>

    <div v-if="store.error" class="col-span-full rounded border border-red-900 bg-red-950/40 p-2 text-sm text-red-300">
      {{ store.error }}
    </div>
  </div>
</template>
