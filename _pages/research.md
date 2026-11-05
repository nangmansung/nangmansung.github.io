---
layout: page
title: Research
permalink: /research/
description: 연구 주제와 진행 중인 프로젝트
nav: true
nav_order: 2
---

{% assign items = site.data.research_pages %}
{% if items and items.size > 0 %}

<div class="research-list">
  {% for item in items %}
    <a class="research-card" href="{{ item.url | relative_url }}">
      <h3>{{ item.title }}</h3>
      {% if item.description %}<p>{{ item.description }}</p>{% endif %}
      {% if item.date != "" %}<span class="research-date">{{ item.date }}</span>{% endif %}
    </a>
  {% endfor %}
</div>

{% else %}
아직 올라온 연구 페이지가 없어요.
{% endif %}
