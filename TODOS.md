# TODOS

## 图像质量评估 Web 管理后台

### TODO-1: 检查 `render_multi_report` 生成的 HTML 是否为自包含

- **What:** 验证报告 HTML 是否内联了所有 CSS/JS，还是引用了外部相对路径文件。
- **Why:** 如果引用了外部文件，FastAPI 的 `/reports/{id}` 直接返回 HTML 会导致样式 404，iframe 中报告显示损坏。
- **Pros:** 避免实现后期返工；确保报告在 iframe 中渲染完整。
- **Cons:** 若报告非自包含，需额外实现静态文件服务或修改报告生成器。
- **Context:** 报告母版 `report_master.html` 体积 8.7MB，可能包含大量内联资源，但 `render_multi_report` 的具体输出方式需要验证。实现 `/reports/{id}` 接口后，加载一次真实报告即可确认。
- **Blocked by:** 需先实现 `/reports/{id}` 接口并生成第一份报告。
- **Added by:** /plan-eng-review on 2026-05-19
