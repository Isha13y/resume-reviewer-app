# OpenAI Quota Management Guide

## 🚨 Why Quota Issues Happen

OpenAI's free tier has very strict limitations:
- **3 requests per minute** max
- **40,000 tokens per month** total (roughly 20-30 resume analyses)
- **$5 free credit** that expires after 3 months
- **Payment method required** even for free usage

## ✅ Optimizations Implemented

### 1. **Exponential Backoff Retry Logic**
- Automatically retries failed requests with increasing delays
- Handles rate limiting (429 errors) gracefully
- Up to 5 retries for free tier users

### 2. **Token Usage Optimization**
- **Free Tier Mode**: Reduced max_tokens (1000 for analysis, 800 for improvements)
- **Shorter Prompts**: Concise prompts that use fewer input tokens
- **Content Truncation**: Automatically truncates long resumes to fit limits
- **Limited Feedback**: Focuses on top 3-5 most important suggestions

### 3. **Smart Error Handling**
- Detects quota/billing issues vs other errors
- Provides specific guidance for each type of error
- Shows helpful retry suggestions

### 4. **Usage Monitoring**
- Estimates token usage before analysis
- Shows timing information for transparency
- Provides usage tips and best practices

### 5. **User Guidance**
- Clear explanation of OpenAI limitations
- Tips for avoiding quota issues
- Links to OpenAI usage dashboard

## 🛠️ How to Use the Optimizations

1. **Enable Free Tier Mode**: Check "Optimize for Free Tier" in the sidebar
2. **Keep Resumes Short**: Under 2 pages works best
3. **Wait Between Requests**: App handles this automatically
4. **Add Payment Method**: Even for free usage, this helps avoid restrictions

## 💡 Best Practices for Free Tier

### For Users:
- Process one resume at a time
- Wait 30 seconds between analyses
- Keep job descriptions under 500 characters
- Use specific job roles for better targeting

### For Troubleshooting:
1. **"Rate limit exceeded"**: Wait a few minutes, app will retry automatically
2. **"Insufficient quota"**: Check your OpenAI billing dashboard
3. **"Authentication failed"**: Verify your API key is correct
4. **"Billing issues"**: Add a payment method to your OpenAI account

## 📊 Token Usage Estimates

| Action | Free Tier Tokens | Standard Tokens |
|--------|------------------|-----------------|
| Resume Analysis | ~800-1200 | ~1500-2500 |
| Resume Improvement | ~600-1000 | ~1000-1800 |
| **Total per resume** | **~1400-2200** | **~2500-4300** |

With 40,000 monthly tokens, you can analyze:
- **18-28 resumes** in free tier optimized mode
- **9-16 resumes** in standard mode

## 🔧 Technical Implementation

### Key Features:
```python
# Retry with exponential backoff
def _make_api_call_with_retry(self, messages, max_tokens=None):
    max_retries = 5 if self.optimize_for_free_tier else 3
    for attempt in range(max_retries):
        try:
            # API call with delay between retries
        except rate_limit_error:
            delay = min(base_delay * (2 ** attempt), max_delay)
            time.sleep(delay)
```

### Optimized Prompts:
- **Free Tier**: "HR expert. Analyze resume for ATS compatibility..."
- **Standard**: Full detailed prompts with comprehensive instructions

## 🆘 Still Having Issues?

1. **Check OpenAI Dashboard**: https://platform.openai.com/usage
2. **Add Payment Method**: Even for free tier usage
3. **Consider Paid Plan**: $0.002 per 1K tokens (very affordable)
4. **Use During Off-Peak**: Early morning or late evening
5. **Contact OpenAI Support**: For account-specific issues

## 📈 Upgrade Recommendations

If you frequently hit limits, consider:
- **Pay-as-you-go**: Only pay for what you use (~$0.10-0.20 per resume)
- **Set spending limits**: Control your monthly costs
- **Monitor usage**: Track your consumption patterns

The optimizations make the free tier much more usable, but for regular use, a small paid plan provides a much better experience!
