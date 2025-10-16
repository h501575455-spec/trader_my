import time
import threading
import sys

class Spinner:
    """Animated emoji loader"""
    def __init__(self):
        self.frames = ["⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽", "⣾"]
        self.done = False
        self.thread = None
        self.message = ""
    
    def start(self, message=""):
        """Start animation thread"""
        self.done = False
        self.message = message
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def update_message(self, new_message):
        self.message = new_message
    
    def _animate(self):
        """Animation insider function"""
        i = 0
        while not self.done:
            frame = self.frames[i % len(self.frames)]
            sys.stdout.write(f"\r{frame} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    
    def stop(self, success=True):
        """Stop animation and show termination message"""
        self.done = True
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        
        # Show status of completion
        message = "✅ Successfully connected to database" if success else "❌ Error establishing database connection"
        status = "🔗 Done!" if success else "🔗 Failed!"

        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.write(f"{message}    {status}\n")
        sys.stdout.flush()
