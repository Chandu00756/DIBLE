from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
@dataclass(frozen=True)
class Settings:
 database_url: str=os.getenv('DIBLE_DATABASE_URL','sqlite:///./data/dible.db')
 environment: str=os.getenv('DIBLE_ENV','development')
 audit_retention_days: int=int(os.getenv('DIBLE_AUDIT_RETENTION_DAYS','365'))
 max_page_size: int=int(os.getenv('DIBLE_MAX_PAGE_SIZE','100'))
 @property
 def sqlite_path(self)->Path:
  if not self.database_url.startswith('sqlite:///'): raise ValueError('this local build supports sqlite URLs only')
  return Path(self.database_url.removeprefix('sqlite:///'))
