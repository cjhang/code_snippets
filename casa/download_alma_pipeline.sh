#!/usr/bin/env bash
set -euo pipefail

LINKS=("
https://casa.nrao.edu/download/distro/casa-pipeline/release/linux/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8.tar.xz
https://casa.nrao.edu/download/distro/casa-pipeline/release/linux/casa-6.6.1-17-pipeline-2024.1.0.8-py3.8.el8.tar.xz
https://casa.nrao.edu/download/distro/casa-pipeline/release/linux/casa-6.5.4-9-pipeline-2023.1.0.124-py3.8.tar.xz
https://casa.nrao.edu/download/distro/casa-pipeline/release/linux/casa-6.4.1-12-pipeline-2022.2.0.68-py3.6.tar.xz
https://casa.nrao.edu/download/distro/casa-pipeline/release/linux/casa-6.4.1-12-pipeline-2022.2.0.64-py3.6.tar.xz
https://casa.nrao.edu/download/distro/casa-pipeline/release/linux/casa-6.2.1-7-pipeline-2021.2.0.128.tar.xz
https://casa.nrao.edu/download/distro/casa-pipeline/release/linux/casa-6.1.1-15-pipeline-2020.1.0.40.tar.xz
https://casa.nrao.edu/download/distro/casa-pipeline/release/el7/casa-pipeline-release-5.6.1-8.el7.tar.gz
https://casa.nrao.edu/download/distro/casa/release/el7/casa-release-5.4.0-70.el7.tar.gz
https://casa.nrao.edu/download/distro/linux/release/el7/casa-release-5.1.1-5.el7.tar.gz
https://casa.nrao.edu/download/distro/linux/release/el7/casa-release-4.7.2-el7.tar.gz
https://casa.nrao.edu/download/distro/linux/release/el7/casa-release-4.7.0-1-el7.tar.gz
https://casa.nrao.edu/download/distro/linux/release/el6/casa-release-4.5.3-el6.tar.gz
https://casa.nrao.edu/download/distro/linux/release/el6/casa-release-4.5.2-el6.tar.gz
https://casa.nrao.edu/download/distro/casa/release/el6/casa-release-4.5.1-el6.tar.gz
https://casa.nrao.edu/download/distro/casa/release/el6/casa-release-4.3.1-pipe-1-el6.tar.gz
https://casa.nrao.edu/download/distro/linux/old/casapy-42.2.30986-pipe-1-64b.tar.gz
")

#for url in "${LINKS[@]}"; do
for url in ${LINKS}; do
  file=$(basename "$url")
  md5file="${file}.md5"

  if [[ -f "$file" ]]; then
    echo "File $file already exists, skipping download."
  else
    echo "Downloading $file..."
    curl -fLO "$url"
  fi

  if [[ -f "$md5file" ]]; then
    echo "MD5 file $md5file already exists, skipping download."
  else
    echo "Downloading $md5file..."
    curl -fLO "${url}.md5"
  fi

  echo "Verifying MD5 for $file..."
  md5sum -c "$md5file"

  #echo "Extracting $file..."
  #tar -xf "$file"

  echo "Done with $file"
  echo "-------------------------"
done

echo "All files processed successfully."
