import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, font
from tkinter import ttk

# ── DeepSeek core imports ────────────────────────────────────────────
from deepseek_chat import (
    add_to_memory,
    retrieve_from_memory,
    is_personal_preference_query,
    messages,
    prune_history,
    client,
    add_image_to_memory,
    add_audio_to_memory
)

# ─────────────────────────────────────────────────────────────────────
class DeepSeekChatGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._configure_root()

        # Top row: entry + buttons
        top = tk.Frame(root, bg="#f0f0f0")
        top.pack(fill=tk.X, padx=14, pady=12)

        entry_font   = font.Font(family="Segoe UI", size=20)
        button_font  = font.Font(family="Segoe UI", size=20, weight="bold")
        chat_font    = font.Font(family="Consolas", size=20)

        self.user_entry = tk.Entry(top, font=entry_font)
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_opts = {"font": button_font, "padx":14, "pady":6}
        tk.Button(top, text="Send", command=self.handle_send, **btn_opts)\
          .pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="Upload Image", command=self.handle_image_upload, **btn_opts)\
          .pack(side=tk.LEFT)
        tk.Button(top, text="Upload Audio", command=self.handle_audio_upload, **btn_opts)\
          .pack(side=tk.LEFT, padx=6)

        # Chat transcript
        self.chat = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            font=chat_font,
            width=1200,
            height=1200
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0,12))
        self.chat.insert(tk.END, "System: " + messages[0]["content"] + "\n\n")

        self.user_entry.focus()

    # ──────────────────────────────────────────
    def _configure_root(self):
        self.root.title("DeepSeek ‑ Multimodal Chat")
        w, h = 1200, 1200
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(900, 900)
        # center
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    # ──────────────────────────────────────────
    def append_chat(self, text: str):
        self.chat.insert(tk.END, text)
        self.chat.see(tk.END)

    # ──────────────────────────────────────────
    def handle_send(self):
        text = self.user_entry.get().strip()
        if not text:
            return
        self.user_entry.delete(0, tk.END)
        self.append_chat(f"User: {text}\n")
        self.process_user_message(text)

    # ──────────────────────────────────────────
    def handle_image_upload(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not path:
            return
        self.append_chat("[Image selected – analysing…]\n")

        def worker():
            desc = add_image_to_memory(path, prompt="" \
            " this image in detail.")
            self.append_chat(f"[Image analysis] {desc}\n")
            self.process_user_message(f"The image shows: {desc}")

        threading.Thread(target=worker, daemon=True).start()
    
    def handle_audio_upload(self):
        path = filedialog.askopenfilename(
            title="Select audio",
            filetypes=[("Audio files", "*.wav *.mp3 *.mp4 *.m4a *.flac *.webm")]
        )
        if not path:
            return
        self.append_chat("[Audio selected – transcribing…]\n")

        def worker():
            transcript = add_audio_to_memory(path)
            self.append_chat(f"[Audio transcript] {transcript}\n")
            self.process_user_message(f"The audio says: {transcript}")

        threading.Thread(target=worker, daemon=True).start()

    # ──────────────────────────────────────────
    def process_user_message(self, user_text: str):
        if is_personal_preference_query(user_text):
            user_text += " (based on my preferences)"

        top_matches = retrieve_from_memory(user_text)[:5]
        ctx = "\n".join(f"Relevant Memory: {e['text']} [tags={e['tags']}]" for e in top_matches)
        messages.append({"role": "user", "content": f"{ctx}\n\nUser: {user_text}"})

        pruned = prune_history(messages)
        if len(pruned) < len(messages):
            messages.clear(); messages.extend(pruned)

        def chat_worker():
            try:
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True
                )
                reply = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta: reply += delta
                messages.append({"role": "assistant", "content": reply})
                add_to_memory(reply, source="assistant", user_id="gui_user")
                self.append_chat(f"Assistant: {reply}\n\n")
            except Exception as e:
                self.append_chat(f"[DeepSeek error] {e}\n")

        threading.Thread(target=chat_worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    DeepSeekChatGUI(root)
    root.mainloop()