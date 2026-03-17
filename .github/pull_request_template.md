## 📋 Description

<!-- What does this PR do? Why is it needed? -->

## 🔗 Related Issue

Closes #<!-- issue number -->

## 🔄 Type of Change

- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 🏋️ Model / training improvement
- [ ] 🔒 Security fix
- [ ] 📷 Camera / hardware integration
- [ ] 📄 Documentation update
- [ ] 🔧 Refactor / cleanup

## 🧪 Testing Done

<!-- How was this tested? What scenarios were verified? -->

- [ ] Tested with video file (`cali_traffic.mp4` or similar)
- [ ] Tested with live RTSP stream
- [ ] Verified Colombian car plates (`AAA123`) are correctly read
- [ ] Verified Colombian motorcycle plates (`AAA12A`) are correctly read
- [ ] No false positives on full-vehicle detections (taxi/bus body not flagged as plate)
- [ ] GPU ran without crashes on RTX 4080 Laptop (or equivalent)

## 📸 Screenshots / Demo

<!-- Add a screenshot or short video clip of the system working if relevant -->

## 🔒 Security Checklist

- [ ] No hardcoded credentials, API keys, or passwords
- [ ] No `.pt` model files committed (add to `.gitignore` if needed)
- [ ] No `.db` database files committed
- [ ] Ran `bandit` scan: 0 HIGH/MEDIUM findings in changed files

## ✅ PR Checklist

- [ ] Code follows existing style (PEP8 for Python)
- [ ] `README.md` updated if new features or setup steps added
- [ ] Branch is up to date with `main`
