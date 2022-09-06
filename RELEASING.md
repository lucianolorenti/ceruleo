# Releasing a new version

Publication in pypi is made automatically by a GitHub action. The action is executed when a tag starting with "v" is pushed to main.
The tag must correspond to the version of `bumpversion.cfg`.

```bash 
git tag -a "v2.0.0" -m "Description of the release"
```


After the publication in pypi, a tag is created and a release is uploaded to github.