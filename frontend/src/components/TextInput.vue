<!-- src/components/TextInput.vue -->
<template>
  <div class="space-y-3">
    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
      Masukkan teks bahasa Indonesia
    </label>
    <textarea
      v-model="localText"
      @input="$emit('update:modelValue', localText)"
      rows="5"
      placeholder="Ketik atau tempel teks di sini... (min. 3 kata)"
      class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 px-4 py-3 text-sm
             focus:outline-none focus:ring-2 focus:ring-teal-500
             focus:border-transparent resize-none transition"
    />
    <div class="flex items-center justify-between">
      <span class="text-xs text-gray-400">{{ localText.length }} karakter</span>
      <div class="flex gap-3">
        <button @click="loadExample"
                type="button"
                class="text-xs text-teal-600 dark:text-teal-400 hover:underline focus:outline-none">
          Contoh teks
        </button>
        <button @click="clearText"
                type="button"
                class="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 focus:outline-none">
          Hapus
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props  = defineProps({ modelValue: String })
const emit   = defineEmits(['update:modelValue'])
const localText = ref(props.modelValue || '')

watch(() => props.modelValue, (val) => { localText.value = val || '' })

const examples = [
  "Selamat pagi, semoga harimu penuh kebaikan dan semangat!",
  "Dasar bodoh, nggak bisa ngerjain hal paling sepele sekalipun!",
  "Semua orang dari kelompok itu tidak layak ada di negeri ini.",
  "Cuaca hari ini cerah banget, cocok buat jalan-jalan ke taman.",
  "Gue beneran muak sama kelakuan orang-orang kayak gitu, bajingan semua!",
]

function loadExample() {
  const idx = Math.floor(Math.random() * examples.length)
  localText.value = examples[idx]
  emit('update:modelValue', localText.value)
}

function clearText() {
  localText.value = ''
  emit('update:modelValue', '')
}
</script>
