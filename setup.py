#!/usr/bin/env python3
from setuptools import setup

PLUGIN_ENTRY_POINT = 'vosk = neon_stt_plugin_vosk_streaming:VoskKaldiStreamingSTT'
setup(
    name='neon-stt-plugin-vosk',
    version='0.2.1',
    description='A vosk stt plugin for mycroft',
    url='https://github.com/NeonGeckoCom/neon-stt-plugin-vosk',
    author='NeonDaniel',
    author_email='developers@neon.ai',
    license='Apache-2.0',
    packages=['neon_stt_plugin_vosk_streaming'],
    install_requires=["numpy", "vosk",
                      "ovos-plugin-manager>=0.0.1a2",
                      "ovos_skill_installer"],
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='mycroft plugin stt',
    entry_points={'mycroft.plugin.stt': PLUGIN_ENTRY_POINT}
)
