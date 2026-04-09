---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

<p style="font-size: 0.92rem; color: #64748b; margin-bottom: 0.5rem;">
  * denotes equal contribution, † denotes corresponding author.<br>
  See also my <a href="https://scholar.google.com/citations?user=vi3W-m8AAAAJ" style="color: #2563eb;">Google Scholar profile</a> for a complete list.
</p>

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-pub.html %}
{% endfor %}
