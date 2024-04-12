#!/bin/bash

rm -rf docs-data
mkdir docs-data
cd docs-data
curl -L https://github.com/CSCfi/csc-user-guide/archive/refs/heads/master.tar.gz -o csc-user-guide.tar.gz
tar -xzf csc-user-guide.tar.gz
rm -f csc-user-guide.tar.gz
mkdir -p csc-user-guide
cd csc-user-guide-master
rm -rf docs/img
cp -r docs ../csc-user-guide
cd ../
rm -rf csc-user-guide-master

curl -L https://github.com/csc-training/csc-env-eff/archive/refs/heads/master.tar.gz -o csc-env-eff.tar.gz
tar -xzf csc-env-eff.tar.gz
rm -f csc-env-eff.tar.gz
mkdir -p csc-env-eff
cd csc-env-eff-master
for f in "part-1" "part-2" "_slides"; do
    cp -r "$f" ../csc-env-eff
done
cd ../
rm -rf csc-env-eff-master

mv csc-env-eff/_slides csc-env-eff/slides
rm -rf csc-env-eff/slides/SRTFiles csc-env-eff/slides/SRTCode csc-env-eff/slides/img csc-env-eff/slides/Makefile
find "csc-env-eff" -name "README.md" -type f -delete
find "csc-env-eff" -name "index.md" -type f -delete

curl -L https://github.com/Lumi-supercomputer/lumi-userguide/archive/refs/heads/production.tar.gz -o lumi-userguide.tar.gz
tar -xzf lumi-userguide.tar.gz
rm -f lumi-userguide.tar.gz
mkdir -p lumi-userguide
cd lumi-userguide-production
cp -r docs/* ../lumi-userguide
cd ../
rm -rf lumi-userguide-production
rm -rf lumi-userguide/assets

cd ../

rm -rf html-data
mkdir html-data
cd html-data
curl -L https://github.com/Lumi-supercomputer/LUMI-EasyBuild-docs/archive/refs/heads/gh-pages.tar.gz -o LUMI-EasyBuild-docs.tar.gz
tar -xzf LUMI-EasyBuild-docs.tar.gz
rm -f LUMI-EasyBuild-docs.tar.gz
mv LUMI-EasyBuild-docs-gh-pages LUMI-EasyBuild-docs
cd LUMI-EasyBuild-docs
rm -rf assets 404.html sitemap.xml* .nojekyll search stylesheets
