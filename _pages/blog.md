---
layout: profile
title: "Blog"
permalink: /blog/
author_profile: false
description: "Research notes, small discoveries, and ideas in progress by Haodong Duan."
---

{% include blog-posts.html %}
<section class="blog-index" aria-labelledby="blog-index-title">
  <header class="blog-index-header">
    <h1 id="blog-index-title">Blog</h1>
  </header>

  <div class="blog-list">
    {% for post in blog_posts %}{% include blog-card.html post=post %}{% else %}<p>More stories soon. Thanks for stopping by.</p>{% endfor %}
  </div>
</section>
