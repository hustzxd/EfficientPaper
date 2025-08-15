// Prism.js 语法定义 for protobuf text format (.prototxt)
Prism.languages.prototxt = {
    'comment': {
        pattern: /#.*$/m,
        greedy: true
    },
    'string': {
        pattern: /"(?:[^"\\]|\\.)*"/,
        greedy: true
    },
    'number': {
        pattern: /\b(?:0x[\da-f]+|[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b/i,
        alias: 'number'
    },
    'boolean': {
        pattern: /\b(?:true|false)\b/i,
        alias: 'boolean'
    },
    'message-name': {
        pattern: /^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*\{)/m,
        lookbehind: true,
        alias: 'class-name'
    },
    'field-name': {
        pattern: /^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*:)/m,
        lookbehind: true,
        alias: 'property'
    },
    'url': {
        pattern: /"(?:https?:\/\/|mailto:)[^"]*"/,
        alias: 'url'
    },
    'identifier': {
        pattern: /\b[a-zA-Z_][a-zA-Z0-9_]*\b/,
        alias: 'variable'
    },
    'punctuation': /[{}[\]:,;]/,
    'operator': /[:=]/
};

// 确保在 Prism 加载后立即定义语言
if (typeof Prism !== 'undefined') {
    Prism.languages.prototxt = Prism.languages.prototxt;
}