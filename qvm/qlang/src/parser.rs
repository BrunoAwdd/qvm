use std::fmt;

use super::{
    aliases::resolve_alias,
    ast::{Expression, Operator, QLangCommand, TypeAnnotation},
};

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Identifier(String),
    Number(f64),
    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,
    Comma,
    Colon,
    Arrow,
    Eq,
    EqEq,
    BangEq,
    Lt, Gt, LtEq, GtEq,
    Plus, Minus, Star, Slash,
    And, Or,
    If, QIf, Else,
    Let, Fn, Import,
    While, For, In, Range,
    Return, Break, Continue,
    Run, Reset,
    MeasureAll, Display,
    Eof,
}

#[derive(Clone)]
pub struct Lexer {
    input: String,
}

impl Lexer {
    pub fn new(input: &str) -> Self { Self { input: input.to_string() } }

    pub fn tokenize(&self) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut chars = self.input.chars().peekable();

        while let Some(&c) = chars.peek() {
            match c {
                ' ' | '\t' | '\n' | '\r' => { chars.next(); }
                '(' => { tokens.push(Token::LParen);   chars.next(); }
                ')' => { tokens.push(Token::RParen);   chars.next(); }
                '{' => { tokens.push(Token::LBrace);   chars.next(); }
                '}' => { tokens.push(Token::RBrace);   chars.next(); }
                '[' => { tokens.push(Token::LBracket); chars.next(); }
                ']' => { tokens.push(Token::RBracket); chars.next(); }
                ',' => { tokens.push(Token::Comma);    chars.next(); }
                '+' => { tokens.push(Token::Plus);     chars.next(); }
                '*' => { tokens.push(Token::Star);     chars.next(); }
                ':' => { tokens.push(Token::Colon);    chars.next(); }
                '-' => {
                    chars.next();
                    if let Some(&'>') = chars.peek() {
                        tokens.push(Token::Arrow);
                        chars.next();
                    } else {
                        tokens.push(Token::Minus);
                    }
                }
                '/' => {
                    chars.next();
                    if let Some(&'/') = chars.peek() {
                        while let Some(&c) = chars.peek() {
                            if c == '\n' { break; }
                            chars.next();
                        }
                    } else {
                        tokens.push(Token::Slash);
                    }
                }
                '=' => {
                    chars.next();
                    if let Some(&'=') = chars.peek() { tokens.push(Token::EqEq); chars.next(); }
                    else { tokens.push(Token::Eq); }
                }
                '!' => {
                    chars.next();
                    if let Some(&'=') = chars.peek() { tokens.push(Token::BangEq); chars.next(); }
                    else { panic!("Unexpected character '!'"); }
                }
                '<' => {
                    chars.next();
                    if let Some(&'=') = chars.peek() { tokens.push(Token::LtEq); chars.next(); }
                    else { tokens.push(Token::Lt); }
                }
                '>' => {
                    chars.next();
                    if let Some(&'=') = chars.peek() { tokens.push(Token::GtEq); chars.next(); }
                    else { tokens.push(Token::Gt); }
                }
                '&' => {
                    chars.next();
                    if let Some(&'&') = chars.peek() { tokens.push(Token::And); chars.next(); }
                    else { panic!("Expected &&"); }
                }
                '|' => {
                    chars.next();
                    if let Some(&'|') = chars.peek() { tokens.push(Token::Or); chars.next(); }
                    else { panic!("Expected ||"); }
                }
                _ if c.is_alphabetic() || c == '_' => {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' { ident.push(c); chars.next(); }
                        else { break; }
                    }
                    tokens.push(match ident.as_str() {
                        "if"         => Token::If,
                        "qif"        => Token::QIf,
                        "else"       => Token::Else,
                        "let"        => Token::Let,
                        "fn"         => Token::Fn,
                        "import"     => Token::Import,
                        "while"      => Token::While,
                        "for"        => Token::For,
                        "in"         => Token::In,
                        "return"     => Token::Return,
                        "break"      => Token::Break,
                        "continue"   => Token::Continue,
                        "run"        => Token::Run,
                        "reset"      => Token::Reset,
                        "measure_all"=> Token::MeasureAll,
                        "display"    => Token::Display,
                        _            => Token::Identifier(ident),
                    });
                }
                _ if c.is_ascii_digit() => {
                    let mut num = String::new();
                    let mut has_dot = false;
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_digit() {
                            num.push(c); chars.next();
                        } else if c == '.' {
                            let mut clone = chars.clone();
                            clone.next();
                            if clone.peek() == Some(&'.') { break; }
                            if !has_dot { has_dot = true; num.push(c); chars.next(); }
                            else { break; }
                        } else { break; }
                    }
                    tokens.push(Token::Number(num.parse().expect("invalid number")));
                }
                '.' => {
                    chars.next();
                    if let Some(&'.') = chars.peek() { chars.next(); tokens.push(Token::Range); }
                    else { panic!("Unexpected '.'"); }
                }
                _ => { chars.next(); }
            }
        }
        tokens.push(Token::Eof);
        tokens
    }
}

#[derive(Clone, Debug)]
pub enum QLangLine {
    Command(QLangCommand),
    Run,
    Reset,
}

impl fmt::Display for QLangLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLangLine::Command(cmd) => write!(f, "{cmd}"),
            QLangLine::Run   => write!(f, "RUN"),
            QLangLine::Reset => write!(f, "RESET"),
        }
    }
}

#[derive(Clone)]
pub struct QLangParser {
    input_buffer: String,
    errors: Vec<String>,
    parsed_commands: Vec<QLangLine>,
}

impl QLangParser {
    pub fn new() -> Self {
        Self { input_buffer: String::new(), errors: vec![], parsed_commands: vec![] }
    }
    pub fn append(&mut self, line: &str) {
        self.input_buffer.push_str(line);
        self.input_buffer.push('\n');
    }
    pub fn validate_lines(&mut self) {
        self.errors.clear();
        self.parsed_commands.clear();
        let tokens = Lexer::new(&self.input_buffer).tokenize();
        let mut p = ParserInternal::new(tokens);
        match p.parse() {
            Ok(cmds) => self.parsed_commands = cmds,
            Err(e)   => self.errors.push(e),
        }
    }
    pub fn has_errors(&self) -> bool { !self.errors.is_empty() }
    pub fn get_errors(&self) -> &[String] { &self.errors }
    pub fn get_commands(&self) -> &[QLangLine] { &self.parsed_commands }
}

struct ParserInternal {
    tokens: Vec<Token>,
    current: usize,
}

impl ParserInternal {
    fn new(tokens: Vec<Token>) -> Self { Self { tokens, current: 0 } }

    fn parse(&mut self) -> Result<Vec<QLangLine>, String> {
        let mut out = Vec::new();
        while !self.is_at_end() { out.push(self.parse_statement()?); }
        Ok(out)
    }

    fn parse_statement(&mut self) -> Result<QLangLine, String> {
        if self.match_token(&[Token::Run])   { return Ok(QLangLine::Run); }
        if self.match_token(&[Token::Reset]) { return Ok(QLangLine::Reset); }
        Ok(QLangLine::Command(self.parse_command()?))
    }

    fn parse_command(&mut self) -> Result<QLangCommand, String> {
        if self.match_token(&[Token::Let])      { return self.parse_let(); }
        if self.match_token(&[Token::Fn])       { return self.parse_function_def(); }
        if self.match_token(&[Token::Import])   { return self.parse_import(); }
        if self.match_token(&[Token::If])       { return self.parse_if(); }
        if self.match_token(&[Token::QIf])      { return self.parse_qif(); }
        if self.match_token(&[Token::While])    { return self.parse_while(); }
        if self.match_token(&[Token::For])      { return self.parse_for(); }
        if self.match_token(&[Token::Return])   {
            return Ok(QLangCommand::Return(self.parse_expression()?));
        }
        if self.match_token(&[Token::Break])    { return Ok(QLangCommand::Break); }
        if self.match_token(&[Token::Continue]) { return Ok(QLangCommand::Continue); }
        if self.match_token(&[Token::MeasureAll]) {
            self.consume(Token::LParen, "Expected '('")?;
            self.consume(Token::RParen, "Expected ')'")?;
            return Ok(QLangCommand::MeasureAll);
        }
        if self.match_token(&[Token::Display]) {
            self.consume(Token::LParen, "Expected '('")?;
            self.consume(Token::RParen, "Expected ')'")?;
            return Ok(QLangCommand::Display);
        }
        if let Token::Identifier(name) = self.peek().clone() {
            self.advance();
            if self.match_token(&[Token::Eq]) {
                return Ok(QLangCommand::Assign { name, value: self.parse_expression()? });
            }
            if self.match_token(&[Token::LParen]) {
                return self.finish_call_statement(name);
            }
        }
        Err(format!("Unexpected token: {:?}", self.peek()))
    }

    fn finish_call_statement(&mut self, name: String) -> Result<QLangCommand, String> {
        if name == "list_functions" {
            self.consume(Token::RParen, "Expected ')'")?;
            return Ok(QLangCommand::ListFunctions);
        }
        if name == "create" {
            let n = self.parse_expression()?;
            self.consume(Token::RParen, "Expected ')'")?;
            return match n {
                Expression::Number(v) => Ok(QLangCommand::Create(v as usize)),
                _ => Err("create() expects a number literal".into()),
            };
        }
        if name == "measure" {
            let mut indices = Vec::new();
            if !self.check(Token::RParen) {
                loop {
                    match self.parse_expression()? {
                        Expression::Number(v) => indices.push(v as usize),
                        _ => return Err("measure() expects number literals for qubit indices".into()),
                    }
                    if !self.match_token(&[Token::Comma]) { break; }
                }
            }
            self.consume(Token::RParen, "Expected ')'")?;
            return Ok(match indices.len() {
                0 => QLangCommand::MeasureAll,
                1 => QLangCommand::Measure(indices[0]),
                _ => QLangCommand::MeasureMany(indices),
            });
        }
        let mut args = Vec::new();
        if !self.check(Token::RParen) {
            loop {
                args.push(self.parse_expression()?);
                if !self.match_token(&[Token::Comma]) { break; }
            }
        }
        self.consume(Token::RParen, "Expected ')'")?;
        let canonical = resolve_alias(&name).to_string();
        Ok(QLangCommand::ApplyGate(canonical, args))
    }

    fn parse_let(&mut self) -> Result<QLangCommand, String> {
        let name = self.expect_identifier("Expected variable name after 'let'")?;
        let type_ann = if self.match_token(&[Token::Colon]) {
            Some(self.parse_type()?)
        } else { None };
        self.consume(Token::Eq, "Expected '=' after variable name")?;
        let value = self.parse_expression()?;
        Ok(QLangCommand::Let { name, type_ann, value })
    }

    fn parse_function_def(&mut self) -> Result<QLangCommand, String> {
        let name = self.expect_identifier("Expected function name")?;
        self.consume(Token::LParen, "Expected '(' after function name")?;
        let mut params      = Vec::new();
        let mut param_types = Vec::new();
        if !self.check(Token::RParen) {
            loop {
                let param = self.expect_identifier("Expected parameter name")?;
                let ptype = if self.match_token(&[Token::Colon]) {
                    Some(self.parse_type()?)
                } else { None };
                params.push(param);
                param_types.push(ptype);
                if !self.match_token(&[Token::Comma]) { break; }
            }
        }
        self.consume(Token::RParen, "Expected ')' after parameters")?;
        let return_type = if self.match_token(&[Token::Arrow]) {
            Some(self.parse_type()?)
        } else { None };
        self.consume(Token::LBrace, "Expected '{' before function body")?;
        let body = self.parse_block()?;
        Ok(QLangCommand::FunctionDef { name, params, param_types, return_type, body })
    }

    fn parse_import(&mut self) -> Result<QLangCommand, String> {
        let path = self.expect_identifier("Expected import path")?;
        Ok(QLangCommand::Import { path })
    }

    fn parse_type(&mut self) -> Result<TypeAnnotation, String> {
        if self.match_token(&[Token::LBracket]) {
            let inner = self.parse_type()?;
            self.consume(Token::RBracket, "Expected ']' after array element type")?;
            return Ok(TypeAnnotation::Array(Box::new(inner)));
        }
        let name = self.expect_identifier("Expected type name")?;
        match name.as_str() {
            "qubit" => Ok(TypeAnnotation::Qubit),
            "bit"   => Ok(TypeAnnotation::Bit),
            "int"   => Ok(TypeAnnotation::Int),
            "float" => Ok(TypeAnnotation::Float),
            "bool"  => Ok(TypeAnnotation::Bool),
            "void"  => Ok(TypeAnnotation::Void),
            other   => Err(format!("Unknown type '{}'. Valid types: qubit, bit, int, float, bool, void, [T]", other)),
        }
    }

    fn parse_if(&mut self) -> Result<QLangCommand, String> {
        self.consume(Token::LParen, "Expected '(' after 'if'")?;
        let condition = self.parse_expression()?;
        self.consume(Token::RParen, "Expected ')' after condition")?;
        self.consume(Token::LBrace, "Expected '{'")?;
        let then_branch = self.parse_block()?;
        let else_branch = if self.match_token(&[Token::Else]) {
            self.consume(Token::LBrace, "Expected '{' after 'else'")?;
            Some(self.parse_block()?)
        } else { None };
        Ok(QLangCommand::If { condition, then_branch, else_branch })
    }

    fn parse_qif(&mut self) -> Result<QLangCommand, String> {
        self.consume(Token::LParen, "Expected '(' after 'qif'")?;
        let condition = self.parse_expression()?;
        self.consume(Token::RParen, "Expected ')' after condition")?;
        self.consume(Token::LBrace, "Expected '{'")?;
        let then_branch = self.parse_block()?;
        let else_branch = if self.match_token(&[Token::Else]) {
            self.consume(Token::LBrace, "Expected '{' after 'else'")?;
            Some(self.parse_block()?)
        } else { None };
        Ok(QLangCommand::QIf { condition, then_branch, else_branch })
    }

    fn parse_while(&mut self) -> Result<QLangCommand, String> {
        self.consume(Token::LParen, "Expected '(' after 'while'")?;
        let condition = self.parse_expression()?;
        self.consume(Token::RParen, "Expected ')'")?;
        self.consume(Token::LBrace, "Expected '{'")?;
        let body = self.parse_block()?;
        Ok(QLangCommand::While { condition, body })
    }

    fn parse_for(&mut self) -> Result<QLangCommand, String> {
        self.consume(Token::LParen, "Expected '(' after 'for'")?;
        let var   = self.expect_identifier("Expected loop variable")?;
        self.consume(Token::In,    "Expected 'in' after loop variable")?;
        let start = self.parse_expression()?;
        self.consume(Token::Range, "Expected '..'") ?;
        let end   = self.parse_expression()?;
        self.consume(Token::RParen, "Expected ')'")?;
        self.consume(Token::LBrace, "Expected '{'")?;
        let body  = self.parse_block()?;
        Ok(QLangCommand::For { var, start, end, body })
    }

    fn parse_block(&mut self) -> Result<Vec<QLangCommand>, String> {
        let mut cmds = Vec::new();
        while !self.check(Token::RBrace) && !self.is_at_end() {
            cmds.push(self.parse_command()?);
        }
        self.consume(Token::RBrace, "Expected '}' to close block")?;
        Ok(cmds)
    }

    fn parse_expression(&mut self) -> Result<Expression, String> { self.parse_logic_or() }

    fn parse_logic_or(&mut self) -> Result<Expression, String> {
        let mut l = self.parse_logic_and()?;
        while self.match_token(&[Token::Or]) {
            let r = self.parse_logic_and()?;
            l = Expression::BinaryOp { left: Box::new(l), op: Operator::Or, right: Box::new(r) };
        }
        Ok(l)
    }

    fn parse_logic_and(&mut self) -> Result<Expression, String> {
        let mut l = self.parse_equality()?;
        while self.match_token(&[Token::And]) {
            let r = self.parse_equality()?;
            l = Expression::BinaryOp { left: Box::new(l), op: Operator::And, right: Box::new(r) };
        }
        Ok(l)
    }

    fn parse_equality(&mut self) -> Result<Expression, String> {
        let mut l = self.parse_comparison()?;
        while self.match_token(&[Token::EqEq, Token::BangEq]) {
            let op = match self.previous() { Token::EqEq => Operator::Eq, _ => Operator::Neq };
            l = Expression::BinaryOp { left: Box::new(l), op, right: Box::new(self.parse_comparison()?) };
        }
        Ok(l)
    }

    fn parse_comparison(&mut self) -> Result<Expression, String> {
        let mut l = self.parse_term()?;
        while self.match_token(&[Token::Lt, Token::Gt, Token::LtEq, Token::GtEq]) {
            let op = match self.previous() {
                Token::Lt => Operator::Lt, Token::Gt => Operator::Gt,
                Token::LtEq => Operator::Le, _ => Operator::Ge,
            };
            l = Expression::BinaryOp { left: Box::new(l), op, right: Box::new(self.parse_term()?) };
        }
        Ok(l)
    }

    fn parse_term(&mut self) -> Result<Expression, String> {
        let mut l = self.parse_factor()?;
        while self.match_token(&[Token::Plus, Token::Minus]) {
            let op = if matches!(self.previous(), Token::Plus) { Operator::Add } else { Operator::Sub };
            l = Expression::BinaryOp { left: Box::new(l), op, right: Box::new(self.parse_factor()?) };
        }
        Ok(l)
    }

    fn parse_factor(&mut self) -> Result<Expression, String> {
        let mut l = self.parse_unary()?;
        while self.match_token(&[Token::Star, Token::Slash]) {
            let op = if matches!(self.previous(), Token::Star) { Operator::Mul } else { Operator::Div };
            l = Expression::BinaryOp { left: Box::new(l), op, right: Box::new(self.parse_unary()?) };
        }
        Ok(l)
    }

    fn parse_unary(&mut self) -> Result<Expression, String> {
        if self.match_token(&[Token::Minus]) {
            return Ok(Expression::BinaryOp {
                left: Box::new(Expression::Number(0.0)),
                op: Operator::Sub,
                right: Box::new(self.parse_unary()?),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expression, String> {
        let mut expr = if self.match_token(&[Token::Number(0.0)]) {
            match self.previous().clone() {
                Token::Number(n) => Expression::Number(n),
                _ => unreachable!(),
            }
        } else if self.match_token(&[Token::Identifier("".into())]) {
            match self.previous().clone() {
                Token::Identifier(name) => {
                    if name == "measure" {
                        self.consume(Token::LParen, "Expected '(' after measure")?;
                        let e = self.parse_expression()?;
                        self.consume(Token::RParen, "Expected ')'")?;
                        Expression::Measure(Box::new(e))
                    } else if self.match_token(&[Token::LParen]) {
                        let mut args = Vec::new();
                        if !self.check(Token::RParen) {
                            loop {
                                args.push(self.parse_expression()?);
                                if !self.match_token(&[Token::Comma]) { break; }
                            }
                        }
                        self.consume(Token::RParen, "Expected ')'")?;
                        Expression::Call(name, args)
                    } else {
                        Expression::Variable(name)
                    }
                }
                _ => unreachable!(),
            }
        } else if self.match_token(&[Token::LParen]) {
            let e = self.parse_expression()?;
            self.consume(Token::RParen, "Expected ')'")?;
            e
        } else if self.match_token(&[Token::LBracket]) {
            let mut es = Vec::new();
            if !self.check(Token::RBracket) {
                loop {
                    es.push(self.parse_expression()?);
                    if !self.match_token(&[Token::Comma]) { break; }
                }
            }
            self.consume(Token::RBracket, "Expected ']'")?;
            Expression::Array(es)
        } else {
            return Err(format!("Unexpected token in expression: {:?}", self.peek()));
        };
        while self.match_token(&[Token::LBracket]) {
            let idx = self.parse_expression()?;
            self.consume(Token::RBracket, "Expected ']'")?;
            expr = Expression::Index(Box::new(expr), Box::new(idx));
        }
        Ok(expr)
    }

    fn expect_identifier(&mut self, msg: &str) -> Result<String, String> {
        if let Token::Identifier(name) = self.peek().clone() {
            self.advance();
            Ok(name)
        } else {
            Err(msg.to_string())
        }
    }

    fn match_token(&mut self, types: &[Token]) -> bool {
        for t in types {
            if self.check_discriminant(t) { self.advance(); return true; }
        }
        false
    }

    fn check(&self, token: Token) -> bool {
        !self.is_at_end() &&
            std::mem::discriminant(self.peek()) == std::mem::discriminant(&token)
    }

    fn check_discriminant(&self, token: &Token) -> bool {
        !self.is_at_end() &&
            std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
    }

    fn consume(&mut self, token: Token, msg: &str) -> Result<Token, String> {
        if self.check_discriminant(&token) { Ok(self.advance().clone()) }
        else { Err(msg.to_string()) }
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() { self.current += 1; }
        self.previous()
    }

    fn is_at_end(&self) -> bool { matches!(self.peek(), Token::Eof) }
    fn peek(&self) -> &Token { &self.tokens[self.current] }
    fn previous(&self) -> &Token { &self.tokens[self.current - 1] }
}