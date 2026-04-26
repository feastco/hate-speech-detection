<!-- src/components/ConfidenceBar.vue -->
<template>
  <div class="space-y-1">
    <div class="flex justify-between text-xs">
      <span class="text-gray-500 dark:text-gray-400 font-medium">Confidence</span>
      <span class="font-bold" :style="{ color: barColor }">
        {{ (confidence * 100).toFixed(1) }}%
      </span>
    </div>
    <div class="w-full bg-gray-100 dark:bg-gray-700 rounded-full h-2.5 overflow-hidden">
      <div
        class="h-full rounded-full transition-all duration-700 ease-out"
        :style="{ width: animatedWidth, backgroundColor: barColor }"
      />
    </div>
    <p class="text-xs text-gray-400 dark:text-gray-500">
      {{ confidenceLabel }}
    </p>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, watch } from 'vue'

const props = defineProps({
  confidence: { type: Number, required: true },
  label:      { type: String, required: true },
})

const animatedWidth = ref('0%')

const barColor = computed(() => {
  if (props.label === 'Hate Speech') return '#E24B4A'
  if (props.label === 'Abusive')     return '#EF9F27'
  return '#639922'
})

const confidenceLabel = computed(() => {
  const pct = props.confidence * 100
  if (pct >= 80) return 'Tinggi — model sangat yakin'
  if (pct >= 60) return 'Sedang — model cukup yakin'
  return 'Rendah — model kurang yakin, perlu review manual'
})

onMounted(() => {
  // Animasi bar dari 0% ke nilai aktual setelah mount
  setTimeout(() => {
    animatedWidth.value = `${props.confidence * 100}%`
  }, 100)
})

// Re-animate when confidence changes (e.g. new prediction)
watch(() => props.confidence, (newVal) => {
  animatedWidth.value = '0%'
  setTimeout(() => {
    animatedWidth.value = `${newVal * 100}%`
  }, 100)
})
</script>
