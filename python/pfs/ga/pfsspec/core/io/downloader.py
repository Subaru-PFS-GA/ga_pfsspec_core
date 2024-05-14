import os
import re
import requests
from requests.utils import requote_uri
import subprocess

from pfs.ga.pfsspec.core.scripts import Plugin

class Downloader(Plugin):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Downloader):
            self.outdir = None
        else:
            self.outdir = orig.outdir

    def add_subparsers(self, configurations, parser):
        return []

    def add_args(self, parser, config):
        super().add_args(parser, config)

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

    def execute_notebooks(self, script):
        pass

    def wget_download(self, url, outfile, headers=None, resume=False, create_dir=True):
        # Download a file with wget

        outdir, _ = os.path.split(outfile)
        if create_dir and not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        
        cmd = [ 'wget', url ]
        cmd.extend(['--tries=3', '--timeout=1', '--wait=1'])
        cmd.extend(['--no-verbose'])
        cmd.extend(['-O', outfile])
        
        if resume:
            cmd.append('--continue')

        if isinstance(headers, list):
            for h in headers:
                cmd.extend(['--header', h])
        elif isinstance(headers, dict):
            for k, v in headers.items():
                cmd.extend(['--header', f'{k}: {v}'])
        elif headers is not None:
            raise ValueError('Invalid headers type')

        subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if os.path.getsize(outfile) == 0:
            os.remove(outfile)
            return False
        else:
            return True
        
    def http_get(self, url, headers=None):
        # Read a file from an URL using requests, pass in auth token
        return requests.get(requote_uri(url), headers=headers, verify=False)
    
    def parse_href(self, html):
        # Parse all URLs in href attributes from an HTML page
        return re.findall(r'href=[\'"]?([^\'" >]+)', html)
        