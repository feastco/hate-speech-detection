<!-- src/App.vue -->
<template>
  <div
    class="min-h-screen bg-gradient-to-br from-gray-50 to-teal-50/30 dark:from-gray-900 dark:to-gray-800 py-10 px-4"
  >
    <div class="max-w-2xl mx-auto space-y-6">
      <!-- ── Header ────────────────────────────────────────────────────── -->
      <div class="text-center space-y-1">
        <h1
          class="text-3xl font-bold tracking-tight text-gray-900 dark:text-white"
        >
          HateDetect<span class="text-teal-600 dark:text-teal-400">ID</span>
        </h1>
        <p class="text-gray-500 dark:text-gray-400 text-sm">
          Deteksi ujaran kebencian bahasa Indonesia
        </p>
        <!-- Status backend -->
        <div class="flex items-center justify-center gap-2 pt-1">
          <span
            class="inline-block w-2 h-2 rounded-full"
            :class="backendOnline ? 'bg-green-400 animate-pulse' : 'bg-red-400'"
          />
          <span class="text-xs text-gray-400">
            {{ backendOnline ? "Backend online" : "Backend offline" }}
          </span>
        </div>
      </div>

      <!-- ── Input Card ─────────────────────────────────────────────────── -->
      <div
        class="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 space-y-4"
      >
        <div class="space-y-1">
          <label
            for="model-select"
            class="block text-xs font-semibold tracking-wide text-gray-500 dark:text-gray-400 uppercase"
          >
            Model Klasifikasi
          </label>
          <select
            id="model-select"
            v-model="selectedModel"
            :disabled="loading || !backendOnline"
            class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-sm text-gray-700 dark:text-gray-100 px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option
              v-for="model in modelOptions"
              :key="model.value"
              :value="model.value"
            >
              {{ model.label }}
            </option>
          </select>
        </div>

        <TextInput v-model="inputText" />
        <button
          @click="analyze"
          :disabled="!inputText.trim() || loading || !backendOnline"
          class="w-full py-3 rounded-lg text-sm font-semibold text-white bg-teal-600 hover:bg-teal-700 active:bg-teal-800 dark:bg-teal-500 dark:hover:bg-teal-600 dark:active:bg-teal-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors duration-150"
        >
          <span v-if="loading" class="flex items-center justify-center gap-2">
            <svg
              class="animate-spin h-4 w-4"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                class="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="4"
              />
              <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8H4z"
              />
            </svg>
            Menganalisis...
          </span>
          <span v-else>Analisis Teks →</span>
        </button>
      </div>

      <!-- ── Error ──────────────────────────────────────────────────────── -->
      <Transition name="fade">
        <div
          v-if="error"
          class="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-xl px-5 py-4 text-sm text-red-700 dark:text-red-400 flex items-start gap-2"
        >
          <span class="text-red-400 shrink-0 mt-0.5">⚠</span>
          <span>{{ error }}</span>
        </div>
      </Transition>

      <!-- ── Result ─────────────────────────────────────────────────────── -->
      <ResultCard :result="result" />

      <!-- ── History ────────────────────────────────────────────────────── -->
      <div
        v-if="history.length > 0"
        class="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-5"
      >
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-200">
            🕓 Riwayat Prediksi
            <span class="ml-1.5 text-xs font-normal text-gray-400">
              ({{ history.length }}/{{ MAX_HISTORY }} terakhir)
            </span>
          </h2>
          <button
            @click="clearHistory"
            type="button"
            class="text-xs text-gray-400 hover:text-red-500 transition-colors"
          >
            Hapus semua
          </button>
        </div>

        <ul class="divide-y divide-gray-50">
          <li
            v-for="(item, idx) in history"
            :key="idx"
            @click="loadFromHistory(item)"
            class="flex items-center gap-3 py-2.5 px-2 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors group"
          >
            <!-- Badge kecil -->
            <span
              class="shrink-0 px-1.5 py-0.5 rounded text-xs font-bold w-8 text-center"
              :class="badgeClass(item.label)"
            >
              {{
                item.label === "Hate Speech"
                  ? "HS"
                  : item.label === "Abusive"
                    ? "AB"
                    : "OK"
              }}
            </span>

            <!-- Teks -->
            <span
              class="text-sm text-gray-600 dark:text-gray-300 truncate flex-1 group-hover:text-gray-900 dark:group-hover:text-white transition-colors"
            >
              {{ item.text }}
            </span>

            <!-- Confidence -->
            <span class="shrink-0 text-xs text-gray-400 font-mono tabular-nums">
              {{ (item.confidence * 100).toFixed(0) }}%
            </span>

            <!-- Model -->
            <span
              class="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-300 uppercase"
            >
              {{ item.model || "svm" }}
            </span>
          </li>
        </ul>
      </div>

      <!-- ── Footer ─────────────────────────────────────────────────────── -->
      <p class="text-center text-xs text-gray-400 dark:text-gray-500">
        Model aktif: {{ modelDisplayName(selectedModel) }} · Dataset: Ibrohim
        &amp; Budi (2019)
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import TextInput from "./components/TextInput.vue";
import ResultCard from "./components/ResultCard.vue";
import { predictText, checkHealth, getModels } from "./api/axios.js";

// ── Konstanta ────────────────────────────────────────────────────────────────
const HISTORY_KEY = "hatedetect_history";
const MAX_HISTORY = 10;

// ── State ────────────────────────────────────────────────────────────────────
const inputText = ref("");
const result = ref(null);
const loading = ref(false);
const error = ref(null);
const history = ref([]);
const backendOnline = ref(false);
const selectedModel = ref("svm");
const modelOptions = ref([
  { value: "svm", label: "SVM (LinearSVC)" },
  { value: "mnb", label: "MNB (Multinomial NB)" },
  { value: "cnb", label: "CNB (Complement NB)" },
  { value: "indobert", label: "IndoBERT (Fine-Tuned)" },
]);

// ── Mount: load history + cek backend ────────────────────────────────────────
onMounted(async () => {
  // Load history dari localStorage
  try {
    const stored = localStorage.getItem(HISTORY_KEY);
    if (stored) history.value = JSON.parse(stored);
  } catch {
    history.value = [];
  }

  // Cek apakah backend online
  try {
    await checkHealth();
    backendOnline.value = true;

    const modelMeta = await getModels();
    if (
      Array.isArray(modelMeta?.available_models) &&
      modelMeta.available_models.length > 0
    ) {
      const modelLabelMap = {
        svm: "SVM (LinearSVC)",
        mnb: "MNB (Multinomial NB)",
        cnb: "CNB (Complement NB)",
        indobert: "IndoBERT (Fine-Tuned)",
      };

      modelOptions.value = modelMeta.available_models.map((name) => ({
        value: name,
        label: modelLabelMap[name] || name.toUpperCase(),
      }));

      selectedModel.value =
        modelMeta.default_model || modelMeta.available_models[0];
    }
  } catch {
    backendOnline.value = false;
    error.value =
      "Backend tidak dapat dijangkau. Jalankan: cd backend && uvicorn main:app --reload --port 8000";
  }
});

// ── History Helpers ───────────────────────────────────────────────────────────
function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.value));
  } catch {
    /* localStorage penuh — abaikan */
  }
}

function pushToHistory(text, res) {
  // Hapus duplikat
  history.value = history.value.filter(
    (h) => !(h.text === text && h.model === selectedModel.value),
  );
  // Prepend
  history.value.unshift({
    text,
    model: selectedModel.value,
    label: res.label,
    confidence: res.confidence,
  });
  // Batasi
  if (history.value.length > MAX_HISTORY) {
    history.value = history.value.slice(0, MAX_HISTORY);
  }
  saveHistory();
}

function loadFromHistory(item) {
  inputText.value = item.text;
  if (item.model) selectedModel.value = item.model;
  analyze();
}

function clearHistory() {
  history.value = [];
  localStorage.removeItem(HISTORY_KEY);
}

function badgeClass(label) {
  if (label === "Hate Speech") return "bg-red-100 text-red-700";
  if (label === "Abusive") return "bg-amber-100 text-amber-700";
  return "bg-green-100 text-green-700";
}

function modelDisplayName(modelKey) {
  const found = modelOptions.value.find((item) => item.value === modelKey);
  return found?.label || String(modelKey || "").toUpperCase();
}

// ── Analisis Utama ────────────────────────────────────────────────────────────
async function analyze() {
  if (!inputText.value.trim() || loading.value) return;

  loading.value = true;
  error.value = null;
  result.value = null;

  try {
    const res = await predictText(inputText.value, selectedModel.value);
    result.value = res;
    pushToHistory(inputText.value, res);
  } catch (err) {
    const msg = err.response?.data?.detail || err.message;
    error.value = `Gagal mendapatkan prediksi: ${msg}`;
  } finally {
    loading.value = false;
  }
}
</script>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
