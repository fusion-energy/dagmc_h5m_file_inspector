
#assumes anaconda-client and conda-build have been installed
# conda install -y anaconda-client
# conda install -y conda-build
# anaconda login

#cleans up files that get uploaded to the anaconda repo if not deleted
find . -name \*.h5m -type f -delete


rm -rf /tmp/conda-build
mkdir -p /tmp/conda-build
rm -rf /tmp/conda-build

conda-build conda/ -c conda-forge --croot /tmp/conda-build 

# converting using all includes quite a few oxs and linux versions.
# conda convert /tmp/conda-build/linux-64/*.tar.bz2 --platform all  -o /tmp/conda-build

# option for converting package to specified platforms
# currently there is no moab conda install on win-64
platforms=( osx-64 linux-64 )
find /tmp/conda-build/linux-64/ -name *.tar.bz2 | while read file
do
    echo $file
    for platform in "${platforms[@]}"
    do
       conda convert --platform $platform $file  -o /tmp/conda-build/
    done
done

anaconda upload -f /tmp/conda-build/*/*.tar.bz2
