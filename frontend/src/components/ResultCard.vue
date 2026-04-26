<!-- src/components/ResultCard.vue -->
<template>
  <Transition name="slide-up">
    <div v-if="result"
         class="border rounded-xl p-5 space-y-4 bg-white dark:bg-gray-800 shadow-sm"
         :class="cardBorderClass">

      <!-- Label + Badge -->
      <div class="flex items-start gap-3">
        <span class="px-3 py-1 rounded-md text-sm font-bold tracking-wide shrink-0"
              :class="badgeClass">
          {{ result.label }}
        </span>
        <span class="text-sm text-gray-500 dark:text-gray-400 pt-0.5">
          {{ labelDescription }}
        </span>
      </div>

      <!-- Confidence Bar -->
      <ConfidenceBar :confidence="result.confidence" :label="result.label" />

      <!-- Top Terms -->
      <div v-if="result.top_terms && result.top_terms.length">
        <p class="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">
          Kata paling berpengaruh
        </p>
        <div class="flex flex-wrap gap-2">
          <span v-for="term in result.top_terms" :key="term"
                class="px-2 py-1 text-xs rounded bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 font-mono
                       border border-gray-200 dark:border-gray-600">
            {{ term }}
          </span>
        </div>
      </div>

      <!-- Processed Text (collapsible) -->
      <details class="group">
        <summary class="text-xs text-gray-400 dark:text-gray-500 cursor-pointer
                        hover:text-gray-600 dark:hover:text-gray-300 select-none list-none
                        flex items-center gap-1">
          <span class="group-open:rotate-90 transition-transform inline-block">▶</span>
          Teks setelah preprocessing
        </summary>
        <p class="mt-2 text-xs font-mono text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/50 p-3 rounded-lg
                  border border-gray-100 dark:border-gray-700 leading-relaxed">
          {{ result.processed_text || '(terlalu pendek — kurang dari 3 token setelah preprocessing)' }}
        </p>
      </details>

      <!-- Footer: latensi -->
      <p class="text-right text-xs text-gray-300 dark:text-gray-600 font-mono">
        ⏱ {{ result.inference_time_ms }} ms
      </p>
    </div>
  </Transition>
</template>

<script setup>
import { computed } from 'vue'
import ConfidenceBar from './ConfidenceBar.vue'

const props = defineProps({ result: Object })

const badgeClass = computed(() => ({
  'bg-red-100 text-red-700 border border-red-200 dark:bg-red-900/40 dark:text-red-400 dark:border-red-800':       props.result?.label === 'Hate Speech',
  'bg-amber-100 text-amber-700 border border-amber-200 dark:bg-amber-900/40 dark:text-amber-400 dark:border-amber-800': props.result?.label === 'Abusive',
  'bg-green-100 text-green-700 border border-green-200 dark:bg-green-900/40 dark:text-green-400 dark:border-green-800': props.result?.label === 'Normal',
}))

const cardBorderClass = computed(() => ({
  'border-red-200 bg-red-50/30 dark:border-red-800 dark:bg-red-900/10':     props.result?.label === 'Hate Speech',
  'border-amber-200 bg-amber-50/30 dark:border-amber-800 dark:bg-amber-900/10': props.result?.label === 'Abusive',
  'border-green-200 bg-green-50/30 dark:border-green-800 dark:bg-green-900/10': props.result?.label === 'Normal',
}))

const labelDescription = computed(() => {
  const map = {
    'Hate Speech': 'Ujaran kebencian yang menarget identitas atau kelompok tertentu',
    'Abusive':     'Bahasa kasar atau umpatan, tidak menarget identitas tertentu',
    'Normal':      'Konten normal, tidak mengandung ujaran kebencian atau bahasa kasar',
  }
  return map[props.result?.label] || ''
})
</script>

<style scoped>
.slide-up-enter-active {
  transition: all 0.35s ease-out;
}
.slide-up-enter-from {
  opacity: 0;
  transform: translateY(12px);
}
</style>
