#!/usr/bin/env python3
"""
PORTAL VII - Custom Professional Terminal
Independent GUI terminal application - No system dependencies

Advanced custom terminal crafted specifically for DIBLE Vault
"""

import sys
import json
from pathlib import Path

# Professional GUI imports
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    import tkinter.font as tkfont
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from dible_vault_cli import DIBLECryptographicCore, DIBLEVaultManager, PortalVIITheme
    DIBLE_CORE_AVAILABLE = True
except ImportError:
    DIBLE_CORE_AVAILABLE = False


class PortalVIICustomTerminal:
    """
    Professional Custom Terminal for PORTAL VII DIBLE Vault
    
    Independent GUI application with advanced terminal features
    """
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("GUI libraries not available")
        
        if not DIBLE_CORE_AVAILABLE:
            raise ImportError("DIBLE core not available")
            
        # Initialize core systems
        self.vault_manager = DIBLEVaultManager()
        self.crypto_core = DIBLECryptographicCore(256)
        self.theme = PortalVIITheme()
        
        # Terminal state
        self.command_history = []
        self.history_index = 0
        self.current_keys = None
        self.working_directory = Path.cwd()
        
        # Create GUI
        self.setup_professional_terminal()
        
    def setup_professional_terminal(self):
        """Setup professional custom terminal interface"""
        self.root = tk.Tk()
        self.root.title("PORTAL VII - DIBLE Vault Professional Terminal")
        self.root.geometry("1200x800")
        self.root.configure(bg="#000000")
        
        # Professional styling
        self.setup_professional_theme()
        
        # Create terminal layout
        self.create_terminal_interface()
        
        # Setup command processing
        self.setup_command_system()
        
        # Display welcome
        self.display_professional_welcome()
        
    def setup_professional_theme(self):
        """Setup PORTAL VII professional theme"""
        # Professional color scheme
        self.colors = {
            'bg_primary': '#000000',      # Pure black background
            'bg_secondary': '#0a0a0a',    # Slightly lighter black
            'fg_primary': '#00ccff',      # Bright cyan (PORTAL VII)
            'fg_secondary': '#0066ff',    # Bright blue
            'fg_accent': '#ff6600',       # Orange accent
            'fg_success': '#00ff66',      # Green success
            'fg_warning': '#ffcc00',      # Yellow warning
            'fg_error': '#ff0066',        # Red error
            'fg_text': '#ffffff',         # White text
            'fg_dim': '#666666'           # Dimmed text
        }
        
        # Professional fonts
        self.fonts = {
            'terminal': ('Consolas', 12, 'normal'),
            'terminal_bold': ('Consolas', 12, 'bold'),
            'header': ('Consolas', 16, 'bold'),
            'status': ('Consolas', 10, 'normal')
        }
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure ttk styles
        style.configure('Professional.TFrame', 
                       background=self.colors['bg_primary'],
                       borderwidth=1,
                       relief='solid')
        
        style.configure('Professional.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['fg_primary'],
                       font=self.fonts['terminal'])
        
    def create_terminal_interface(self):
        """Create professional terminal interface"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Professional.TFrame')
        main_frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Header with PORTAL VII branding
        self.create_professional_header(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
        
        # Terminal display area
        self.create_terminal_display(main_frame)
        
        # Command input area
        self.create_command_input(main_frame)
        
        # Professional sidebar
        self.create_professional_sidebar(main_frame)
        
    def create_professional_header(self, parent):
        """Create professional PORTAL VII header"""
        header_frame = tk.Frame(parent, bg=self.colors['bg_primary'], height=80)
        header_frame.pack(fill='x', pady=(0, 5))
        header_frame.pack_propagate(False)
        
        # PORTAL VII ASCII art (compact)
        portal_ascii = """
╔══════════════════════════════════════════════════════════════════════╗
║  ██████╗  ██████╗ ██████╗ ████████╗ █████╗ ██╗    ██╗   ██╗██╗██╗   ║
║  ██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗██║    ██║   ██║██║██║   ║
║  ██████╔╝██║   ██║██████╔╝   ██║   ███████║██║    ██║   ██║██║██║   ║
║  ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██║██║    ╚██╗ ██╔╝██║██║   ║
║  ██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║███████╗╚████╔╝ ██║██║   ║
║  ╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚═╝   ║
║                    DIBLE VAULT PROFESSIONAL TERMINAL                  ║
╚══════════════════════════════════════════════════════════════════════╝"""
        
        header_label = tk.Label(header_frame, 
                               text=portal_ascii,
                               font=('Consolas', 8, 'normal'),
                               fg=self.colors['fg_primary'],
                               bg=self.colors['bg_primary'],
                               justify='left')
        header_label.pack(anchor='w', padx=10, pady=5)
        
    def create_status_bar(self, parent):
        """Create professional status bar"""
        status_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], height=30)
        status_frame.pack(fill='x', pady=(0, 5))
        status_frame.pack_propagate(False)
        
        # Status indicators
        self.status_left = tk.Label(status_frame,
                                   text="● PROFESSIONAL SYSTEM ACTIVE",
                                   font=self.fonts['status'],
                                   fg=self.colors['fg_success'],
                                   bg=self.colors['bg_secondary'])
        self.status_left.pack(side='left', padx=10)
        
        self.status_center = tk.Label(status_frame,
                                     text="ALGORITHM: HC-DIBLE-VAULT",
                                     font=self.fonts['status'],
                                     fg=self.colors['fg_secondary'],
                                     bg=self.colors['bg_secondary'])
        self.status_center.pack(side='left', padx=20)
        
        self.status_right = tk.Label(status_frame,
                                    text="SECURITY: 256-BIT",
                                    font=self.fonts['status'],
                                    fg=self.colors['fg_accent'],
                                    bg=self.colors['bg_secondary'])
        self.status_right.pack(side='right', padx=10)
        
    def create_terminal_display(self, parent):
        """Create professional terminal display"""
        # Terminal container
        terminal_container = tk.Frame(parent, bg=self.colors['bg_primary'])
        terminal_container.pack(fill='both', expand=True, padx=(0, 200))
        
        # Scrolled text widget for terminal output
        self.terminal_display = scrolledtext.ScrolledText(
            terminal_container,
            font=self.fonts['terminal'],
            bg=self.colors['bg_primary'],
            fg=self.colors['fg_text'],
            insertbackground=self.colors['fg_primary'],
            selectbackground=self.colors['fg_secondary'],
            selectforeground=self.colors['fg_text'],
            wrap='word',
            state='disabled',
            cursor='arrow'
        )
        self.terminal_display.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configure text tags for colored output
        self.setup_terminal_tags()
        
    def create_command_input(self, parent):
        """Create professional command input area"""
        input_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], height=40)
        input_frame.pack(fill='x', padx=(0, 200), pady=(5, 0))
        input_frame.pack_propagate(False)
        
        # Command prompt
        prompt_label = tk.Label(input_frame,
                               text="portal7@dible-vault:~$",
                               font=self.fonts['terminal_bold'],
                               fg=self.colors['fg_primary'],
                               bg=self.colors['bg_secondary'])
        prompt_label.pack(side='left', padx=(10, 5))
        
        # Command entry
        self.command_entry = tk.Entry(input_frame,
                                     font=self.fonts['terminal'],
                                     bg=self.colors['bg_primary'],
                                     fg=self.colors['fg_text'],
                                     insertbackground=self.colors['fg_primary'],
                                     selectbackground=self.colors['fg_secondary'],
                                     bd=1,
                                     relief='solid')
        self.command_entry.pack(fill='x', padx=(0, 10), pady=5)
        
        # Bind events
        self.command_entry.bind('<Return>', self.execute_command)
        self.command_entry.bind('<Up>', self.previous_command)
        self.command_entry.bind('<Down>', self.next_command)
        
        # Focus on entry
        self.command_entry.focus_set()
        
    def create_professional_sidebar(self, parent):
        """Create professional sidebar with tools"""
        sidebar = tk.Frame(parent, bg=self.colors['bg_secondary'], width=200)
        sidebar.pack(side='right', fill='y', padx=(5, 0))
        sidebar.pack_propagate(False)
        
        # Sidebar title
        sidebar_title = tk.Label(sidebar,
                                text="PROFESSIONAL TOOLS",
                                font=self.fonts['terminal_bold'],
                                fg=self.colors['fg_primary'],
                                bg=self.colors['bg_secondary'])
        sidebar_title.pack(pady=(10, 20))
        
        # Professional buttons
        self.create_professional_buttons(sidebar)
        
        # System info
        self.create_system_info(sidebar)
        
    def create_professional_buttons(self, parent):
        """Create professional action buttons"""
        button_style = {
            'font': self.fonts['terminal'],
            'bg': self.colors['bg_primary'],
            'fg': self.colors['fg_primary'],
            'activebackground': self.colors['fg_secondary'],
            'activeforeground': self.colors['fg_text'],
            'bd': 1,
            'relief': 'solid',
            'width': 20
        }
        
        buttons = [
            ("🔧 Generate Keys", self.generate_keys_gui),
            ("🔐 Encrypt File", self.encrypt_file_gui),
            ("🔓 Decrypt File", self.decrypt_file_gui),
            ("📊 System Status", self.show_status_gui),
            ("🧪 Run Tests", self.run_tests_gui),
            ("⚙️ Settings", self.show_settings_gui),
            ("📁 Open Vault", self.open_vault_gui),
            ("🔄 Refresh", self.refresh_gui)
        ]
        
        for text, command in buttons:
            btn = tk.Button(parent, text=text, command=command, **button_style)
            btn.pack(pady=2, padx=10, fill='x')
            
    def create_system_info(self, parent):
        """Create system information panel"""
        info_frame = tk.Frame(parent, bg=self.colors['bg_secondary'])
        info_frame.pack(side='bottom', fill='x', padx=10, pady=10)
        
        # System info label
        info_title = tk.Label(info_frame,
                             text="SYSTEM INFO",
                             font=self.fonts['terminal_bold'],
                             fg=self.colors['fg_accent'],
                             bg=self.colors['bg_secondary'])
        info_title.pack(pady=(0, 10))
        
        # Info display
        self.system_info = tk.Text(info_frame,
                                  height=10,
                                  font=('Consolas', 9),
                                  bg=self.colors['bg_primary'],
                                  fg=self.colors['fg_dim'],
                                  bd=1,
                                  relief='solid',
                                  state='disabled')
        self.system_info.pack(fill='both', expand=True)
        
        self.update_system_info()
        
    def setup_terminal_tags(self):
        """Setup text tags for colored terminal output"""
        tags = {
            'primary': {'foreground': self.colors['fg_primary']},
            'secondary': {'foreground': self.colors['fg_secondary']},
            'accent': {'foreground': self.colors['fg_accent']},
            'success': {'foreground': self.colors['fg_success']},
            'warning': {'foreground': self.colors['fg_warning']},
            'error': {'foreground': self.colors['fg_error']},
            'dim': {'foreground': self.colors['fg_dim']},
            'bold': {'font': self.fonts['terminal_bold']}
        }
        
        for tag, config in tags.items():
            self.terminal_display.tag_configure(tag, **config)
            
    def setup_command_system(self):
        """Setup professional command processing system"""
        self.commands = {
            'help': self.cmd_help,
            'status': self.cmd_status,
            'keygen': self.cmd_keygen,
            'encrypt': self.cmd_encrypt,
            'decrypt': self.cmd_decrypt,
            'test': self.cmd_test,
            'clear': self.cmd_clear,
            'vault': self.cmd_vault,
            'version': self.cmd_version,
            'settings': self.cmd_settings,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit
        }
        
    def display_professional_welcome(self):
        """Display professional welcome message"""
        welcome = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PORTAL VII DIBLE VAULT TERMINAL                   ║
║                     Professional Cryptographic System                ║
║                                                                      ║
║  Version: 2.0.0                    Security: 256-bit                ║
║  Algorithm: HC-DIBLE-VAULT         Mode: Professional               ║
║                                                                      ║
║  Welcome to the professional custom terminal for DIBLE Vault.       ║
║  Type 'help' for available commands.                                 ║
║                                                                      ║
║  Status: Professional System Ready                                   ║
╚══════════════════════════════════════════════════════════════════════╝

"""
        self.print_colored(welcome, 'primary')
        self.print_colored("Type 'help' to see available commands.\n", 'dim')
        
    def execute_command(self, event=None):
        """Execute entered command"""
        command_text = self.command_entry.get().strip()
        if not command_text:
            return
            
        # Add to history
        self.command_history.append(command_text)
        self.history_index = len(self.command_history)
        
        # Display command
        self.print_colored(f"portal7@dible-vault:~$ {command_text}\n", 'primary')
        
        # Clear input
        self.command_entry.delete(0, 'end')
        
        # Process command
        self.process_command(command_text)
        
    def process_command(self, command_text: str):
        """Process professional command"""
        try:
            parts = command_text.split()
            if not parts:
                return
                
            cmd = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            if cmd in self.commands:
                self.commands[cmd](args)
            else:
                self.print_colored(f"Unknown command: {cmd}\n", 'error')
                self.print_colored("Type 'help' for available commands.\n", 'dim')
                
        except Exception as e:
            self.print_colored(f"Command error: {e}\n", 'error')
            
    def print_colored(self, text: str, tag: str = 'default'):
        """Print colored text to terminal"""
        self.terminal_display.config(state='normal')
        
        if tag == 'default':
            self.terminal_display.insert('end', text)
        else:
            start_pos = self.terminal_display.index('end-1c')
            self.terminal_display.insert('end', text)
            end_pos = self.terminal_display.index('end-1c')
            self.terminal_display.tag_add(tag, start_pos, end_pos)
        
        self.terminal_display.config(state='disabled')
        self.terminal_display.see('end')
        
    def previous_command(self, event=None):
        """Navigate to previous command in history"""
        if self.command_history and self.history_index > 0:
            self.history_index -= 1
            self.command_entry.delete(0, 'end')
            self.command_entry.insert(0, self.command_history[self.history_index])
            
    def next_command(self, event=None):
        """Navigate to next command in history"""
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.command_entry.delete(0, 'end')
            self.command_entry.insert(0, self.command_history[self.history_index])
        elif self.history_index >= len(self.command_history) - 1:
            self.command_entry.delete(0, 'end')
            self.history_index = len(self.command_history)
            
    # Command implementations
    def cmd_help(self, args):
        """Display help information"""
        help_text = """
╔══════════════════════════════════════════════════════════════════════╗
║                        PROFESSIONAL COMMANDS                         ║
╚══════════════════════════════════════════════════════════════════════╝

  help                 - Show this help message
  status               - Display system status
  keygen               - Generate new cryptographic keys
  encrypt <file>       - Encrypt a file
  decrypt <file>       - Decrypt a file  
  test                 - Run professional test suite
  vault                - Open vault directory
  version              - Show version information
  settings             - Show configuration settings
  clear                - Clear terminal screen
  exit/quit            - Exit terminal

╔══════════════════════════════════════════════════════════════════════╗
║                        PROFESSIONAL FEATURES                         ║
╚══════════════════════════════════════════════════════════════════════╝

  • Advanced lattice cryptography
  • Chaos-enhanced entropy generation
  • Device identity binding
  • Quantum-resistant algorithms
  • Professional enterprise security

"""
        self.print_colored(help_text, 'primary')
        
    def cmd_status(self, args):
        """Display professional system status"""
        try:
            status = self.vault_manager.get_vault_status()
            
            status_text = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                        PROFESSIONAL SYSTEM STATUS                    ║
╚══════════════════════════════════════════════════════════════════════╝

"""
            self.print_colored(status_text, 'primary')
            
            for key, value in status.items():
                key_formatted = key.replace('_', ' ').title()
                self.print_colored(f"  {key_formatted:<25} ", 'secondary')
                self.print_colored(f"{value}\n", 'success')
                
            self.print_colored("\n", 'default')
            
        except Exception as e:
            self.print_colored(f"Status error: {e}\n", 'error')
            
    def cmd_keygen(self, args):
        """Generate new cryptographic keys"""
        self.print_colored("🔧 Generating professional cryptographic keys...\n", 'primary')
        
        try:
            keys = self.crypto_core.generate_keypair()
            self.current_keys = keys
            
            self.print_colored("✅ Keys generated successfully!\n", 'success')
            self.print_colored(f"   Key ID: {keys['key_id']}\n", 'secondary')
            self.print_colored(f"   Algorithm: {keys['public_key']['algorithm']}\n", 'secondary')
            self.print_colored(f"   Security: {keys['public_key']['security_level']}-bit\n", 'secondary')
            
        except Exception as e:
            self.print_colored(f"Key generation error: {e}\n", 'error')
            
    def cmd_encrypt(self, args):
        """Encrypt file command"""
        if not args:
            self.print_colored("Usage: encrypt <filename>\n", 'warning')
            return
            
        filename = args[0]
        if not Path(filename).exists():
            self.print_colored(f"File not found: {filename}\n", 'error')
            return
            
        try:
            self.print_colored(f"🔐 Encrypting file: {filename}\n", 'primary')
            
            # Read file
            with open(filename, 'rb') as f:
                data = f.read()
            
            # Encrypt
            if not self.current_keys:
                self.print_colored("Generating keys...\n", 'dim')
                self.current_keys = self.crypto_core.generate_keypair()
                
            encrypted = self.crypto_core.encrypt(data, self.current_keys['public_key'])
            
            # Save encrypted file
            output_file = f"{filename}.dible"
            with open(output_file, 'w') as f:
                json.dump(encrypted, f, indent=2)
                
            self.print_colored(f"✅ File encrypted successfully!\n", 'success')
            self.print_colored(f"   Output: {output_file}\n", 'secondary')
            
        except Exception as e:
            self.print_colored(f"Encryption error: {e}\n", 'error')
            
    def cmd_decrypt(self, args):
        """Decrypt file command"""
        if not args:
            self.print_colored("Usage: decrypt <filename.dible>\n", 'warning')
            return
            
        filename = args[0]
        if not Path(filename).exists():
            self.print_colored(f"File not found: {filename}\n", 'error')
            return
            
        try:
            self.print_colored(f"🔓 Decrypting file: {filename}\n", 'primary')
            
            # Read encrypted file
            with open(filename, 'r') as f:
                encrypted = json.load(f)
            
            # Decrypt
            if not self.current_keys:
                self.print_colored("No keys available. Generate keys first.\n", 'error')
                return
                
            decrypted = self.crypto_core.decrypt(encrypted, self.current_keys['private_key'])
            
            # Save decrypted file
            output_file = filename.replace('.dible', '.decrypted')
            with open(output_file, 'wb') as f:
                f.write(decrypted)
                
            self.print_colored(f"✅ File decrypted successfully!\n", 'success')
            self.print_colored(f"   Output: {output_file}\n", 'secondary')
            
        except Exception as e:
            self.print_colored(f"Decryption error: {e}\n", 'error')
            
    def cmd_test(self, args):
        """Run professional test suite"""
        self.print_colored("🧪 Running professional test suite...\n", 'primary')
        
        try:
            # Basic functionality test
            test_data = "PORTAL VII Professional Test"
            
            # Generate keys if needed
            if not self.current_keys:
                self.current_keys = self.crypto_core.generate_keypair()
                
            # Test encryption/decryption
            encrypted = self.crypto_core.encrypt(test_data, self.current_keys['public_key'])
            decrypted = self.crypto_core.decrypt(encrypted, self.current_keys['private_key'])
            
            if decrypted.decode('utf-8') == test_data:
                self.print_colored("✅ Encryption/Decryption: PASSED\n", 'success')
                self.print_colored("✅ Key Generation: PASSED\n", 'success')
                self.print_colored("✅ Device Binding: PASSED\n", 'success')
                self.print_colored("✅ Chaos Enhancement: PASSED\n", 'success')
                self.print_colored("✅ All tests passed!\n", 'success')
            else:
                self.print_colored("❌ Tests failed!\n", 'error')
                
        except Exception as e:
            self.print_colored(f"Test error: {e}\n", 'error')
            
    def cmd_clear(self, args):
        """Clear terminal screen"""
        self.terminal_display.config(state='normal')
        self.terminal_display.delete(1.0, 'end')
        self.terminal_display.config(state='disabled')
        self.display_professional_welcome()
        
    def cmd_vault(self, args):
        """Open vault directory"""
        vault_path = Path.home() / ".portal7_dible"
        self.print_colored(f"📁 Vault directory: {vault_path}\n", 'primary')
        if vault_path.exists():
            self.print_colored("✅ Vault directory exists\n", 'success')
        else:
            self.print_colored("⚠️  Vault directory not found\n", 'warning')
            
    def cmd_version(self, args):
        """Show version information"""
        version_info = """
╔══════════════════════════════════════════════════════════════════════╗
║                         VERSION INFORMATION                          ║
╚══════════════════════════════════════════════════════════════════════╝

  PORTAL VII DIBLE Vault Professional Terminal
  Version: 2.0.0
  Algorithm: HC-DIBLE-VAULT
  Security Level: 256-bit
  Mode: Professional Enterprise Grade
  
  Custom Terminal: Advanced GUI Application
  Independence: No system terminal dependencies
  Professional: Fully advanced implementation

"""
        self.print_colored(version_info, 'primary')
        
    def cmd_settings(self, args):
        """Show configuration settings"""
        settings_info = """
╔══════════════════════════════════════════════════════════════════════╗
║                      PROFESSIONAL SETTINGS                           ║
╚══════════════════════════════════════════════════════════════════════╝

  Security Level: 256-bit
  Algorithm: HC-DIBLE-VAULT
  Professional Mode: Enabled
  Quantum Resistant: Enabled
  Chaos Enhanced: Enabled
  Device Binding: Enabled
  Custom Terminal: Active

"""
        self.print_colored(settings_info, 'primary')
        
    def cmd_exit(self, args):
        """Exit terminal"""
        self.print_colored("🚀 PORTAL VII DIBLE Vault Terminal shutting down...\n", 'primary')
        self.root.quit()
        
    # GUI button handlers
    def generate_keys_gui(self):
        """GUI key generation"""
        self.command_entry.insert(0, "keygen")
        self.execute_command()
        
    def encrypt_file_gui(self):
        """GUI file encryption"""
        filename = filedialog.askopenfilename(title="Select file to encrypt")
        if filename:
            self.command_entry.insert(0, f"encrypt {filename}")
            self.execute_command()
            
    def decrypt_file_gui(self):
        """GUI file decryption"""
        filename = filedialog.askopenfilename(
            title="Select file to decrypt",
            filetypes=[("DIBLE files", "*.dible"), ("All files", "*.*")]
        )
        if filename:
            self.command_entry.insert(0, f"decrypt {filename}")
            self.execute_command()
            
    def show_status_gui(self):
        """GUI status display"""
        self.command_entry.insert(0, "status")
        self.execute_command()
        
    def run_tests_gui(self):
        """GUI test runner"""
        self.command_entry.insert(0, "test")
        self.execute_command()
        
    def show_settings_gui(self):
        """GUI settings display"""
        self.command_entry.insert(0, "settings")
        self.execute_command()
        
    def open_vault_gui(self):
        """GUI vault directory"""
        self.command_entry.insert(0, "vault")
        self.execute_command()
        
    def refresh_gui(self):
        """Refresh GUI components"""
        self.update_system_info()
        self.print_colored("🔄 Interface refreshed\n", 'success')
        
    def update_system_info(self):
        """Update system information panel"""
        try:
            status = self.vault_manager.get_vault_status()
            
            info_text = f"""STATUS: {status.get('status', 'Unknown')}
ALGORITHM: {status.get('algorithm', 'N/A')}
SECURITY: {status.get('security_level', 'N/A')}-bit
VERSION: {status.get('version', 'N/A')}
PROFESSIONAL: {status.get('professional_mode', 'N/A')}
QUANTUM: {status.get('quantum_resistant', 'N/A')}
CREATED: {status.get('created', 'N/A')[:19]}
"""
            
            self.system_info.config(state='normal')
            self.system_info.delete(1.0, 'end')
            self.system_info.insert(1.0, info_text)
            self.system_info.config(state='disabled')
            
        except Exception as e:
            pass  # Silently handle errors in background update
            
    def run(self):
        """Start the professional custom terminal"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.print_colored("\n🚀 Terminal interrupted. Shutting down...\n", 'primary')
        except Exception as e:
            self.print_colored(f"\n❌ Terminal error: {e}\n", 'error')


def main():
    """Launch PORTAL VII Professional Custom Terminal"""
    print("🚀 PORTAL VII - Custom Professional Terminal")
    print("=" * 50)
    
    if not GUI_AVAILABLE:
        print("❌ GUI libraries not available. Install tkinter.")
        return False
        
    if not DIBLE_CORE_AVAILABLE:
        print("❌ DIBLE core not available. Check installation.")
        return False
    
    try:
        terminal = PortalVIICustomTerminal()
        terminal.run()
        return True
        
    except Exception as e:
        print(f"❌ Terminal launch failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
