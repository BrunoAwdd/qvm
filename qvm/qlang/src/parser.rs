use std::fmt;


use super::{aliases::resolve_alias, ast::{Expression, Operator, QLangCommand}};

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Identifier(String),
    Number(f64),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket, // [
    RBracket, // ]
    Comma,
    Eq,         // =
    EqEq,       // ==
    BangEq,     // !=
    Lt,         // <
    Gt,         // >
    LtEq,       // <=
    GtEq,       // >=
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    And,        // &&
    Or,         // ||
    If,
    QIf,
    Else,
    Let,
    Fn,
    Import,
    While,
    For,
    In,
    Range, // ..
    Run,
    Reset,
    MeasureAll,
    Display,
    Eof,
}

#[derive(Clone)]
pub struct Lexer {
    input: String,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
        }
    }

    pub fn tokenize(&self) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut chars = self.input.chars().peekable();

        while let Some(&c) = chars.peek() {
            match c {
                ' ' | '\t' | '\n' | '\r' => {
                    chars.next();
                }
                '(' => {
                    tokens.push(Token::LParen);
                    chars.next();
                }
                ')' => {
                    tokens.push(Token::RParen);
                    chars.next();
                }
                '{' => {
                    tokens.push(Token::LBrace);
                    chars.next();
                }
                '}' => {
                    tokens.push(Token::RBrace);
                    chars.next();
                }
                '[' => {
                    tokens.push(Token::LBracket);
                    chars.next();
                }
                ']' => {
                    tokens.push(Token::RBracket);
                    chars.next();
                }
                ',' => {
                    tokens.push(Token::Comma);
                    chars.next();
                }
                '+' => {
                    tokens.push(Token::Plus);
                    chars.next();
                }
                '-' => {
                    tokens.push(Token::Minus);
                    chars.next();
                }
                '*' => {
                    tokens.push(Token::Star);
                    chars.next();
                }
                '/' => {
                    // Check for comments //
                    chars.next();
                    if let Some(&'/') = chars.peek() {
                        while let Some(&c) = chars.peek() {
                            if c == '\n' {
                                break;
                            }
                            chars.next();
                        }
                    } else {
                        tokens.push(Token::Slash);
                    }
                }
                '=' => {
                    chars.next();
                    if let Some(&'=') = chars.peek() {
                        tokens.push(Token::EqEq);
                        chars.next();
                    } else {
                        tokens.push(Token::Eq);
                    }
                }
                '!' => {
                    chars.next();
                    if let Some(&'=') = chars.peek() {
                        tokens.push(Token::BangEq);
                        chars.next();
                    } else {
                        // Unexpected single '!'
                        // For now, ignore or panic? Let's panic for simplicity in this phase
                        panic!("Unexpected character '!'");
                    }
                }
                '<' => {
                    chars.next();
                    if let Some(&'=') = chars.peek() {
                        tokens.push(Token::LtEq);
                        chars.next();
                    } else {
                        tokens.push(Token::Lt);
                    }
                }
                '>' => {
                    chars.next();
                    if let Some(&'=') = chars.peek() {
                        tokens.push(Token::GtEq);
                        chars.next();
                    } else {
                        tokens.push(Token::Gt);
                    }
                }
                '&' => {
                    chars.next();
                    if let Some(&'&') = chars.peek() {
                        tokens.push(Token::And);
                        chars.next();
                    } else {
                        panic!("Expected &&");
                    }
                }
                '|' => {
                    chars.next();
                    if let Some(&'|') = chars.peek() {
                        tokens.push(Token::Or);
                        chars.next();
                    } else {
                        panic!("Expected ||");
                    }
                }
                _ if c.is_alphabetic() || c == '_' => {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            ident.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    match ident.as_str() {
                        "if" => tokens.push(Token::If),
                        "qif" => tokens.push(Token::QIf),
                        "else" => tokens.push(Token::Else),
                        "let" => tokens.push(Token::Let),
                        "fn" => tokens.push(Token::Fn),
                        "import" => tokens.push(Token::Import),
                        "while" => tokens.push(Token::While),
                        "for" => tokens.push(Token::For),
                        "in" => tokens.push(Token::In),
                        "run" => tokens.push(Token::Run),
                        "reset" => tokens.push(Token::Reset),
                        "measure_all" => tokens.push(Token::MeasureAll),
                        "display" => tokens.push(Token::Display),
                        _ => tokens.push(Token::Identifier(ident)),
                    }
                }
                _ if c.is_digit(10) => {
                    let mut num_str = String::new();
                    let mut has_dot = false;
                    
                    while let Some(&c) = chars.peek() {
                        if c.is_digit(10) {
                            num_str.push(c);
                            chars.next();
                        } else if c == '.' {
                            // Check if it's a range '..'
                            // If we already have a dot, it's definitely not part of this number (invalid float or range start)
                            // If next char is also '.', it's a range, so stop parsing number.
                            
                            // Peek next char without consuming current dot yet
                            let mut iter_clone = chars.clone();
                            iter_clone.next(); // skip current dot
                            if let Some(&next_c) = iter_clone.peek() {
                                if next_c == '.' {
                                    // It is '..', so stop number parsing here.
                                    break;
                                }
                            }
                            
                            if !has_dot {
                                has_dot = true;
                                num_str.push(c);
                                chars.next();
                            } else {
                                // Second dot in number, stop.
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if let Ok(n) = num_str.parse::<f64>() {
                        tokens.push(Token::Number(n));
                    } else {
                         panic!("Invalid number format: {}", num_str);
                    }
                }
                '.' => {
                    chars.next();
                    if let Some(&'.') = chars.peek() {
                        chars.next();
                        tokens.push(Token::Range);
                    } else {
                         // Just a dot, maybe error or unexpected?
                         // For now let's panic or ignore.
                         panic!("Unexpected character '.'");
                    }
                }
                _ => {
                    // Ignore unknown chars or panic
                    chars.next();
                }
            }
        }
        tokens.push(Token::Eof);
        tokens
    }
}

/// Represents a single parsed line/statement in the QLang program.
/// Kept for compatibility with existing code structure, but extended.
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
            QLangLine::Run => write!(f, "RUN"),
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
        Self {
            input_buffer: String::new(),
            errors: vec![],
            parsed_commands: vec![],
        }
    }

    pub fn append(&mut self, line: &str) {
        self.input_buffer.push_str(line);
        self.input_buffer.push('\n');
    }

    pub fn validate_lines(&mut self) {
        self.errors.clear();
        self.parsed_commands.clear();

        let lexer = Lexer::new(&self.input_buffer);
        let tokens = lexer.tokenize();
        let mut parser = ParserInternal::new(tokens);

        match parser.parse() {
            Ok(commands) => self.parsed_commands = commands,
            Err(e) => self.errors.push(e),
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
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, current: 0 }
    }

    fn parse(&mut self) -> Result<Vec<QLangLine>, String> {
        let mut commands = Vec::new();
        while !self.is_at_end() {
            commands.push(self.parse_statement()?);
        }
        Ok(commands)
    }

    fn parse_statement(&mut self) -> Result<QLangLine, String> {
        println!("Parsing statement. Next token: {:?}", self.peek());
        if self.match_token(&[Token::Run]) {
            return Ok(QLangLine::Run);
        }
        if self.match_token(&[Token::Reset]) {
            return Ok(QLangLine::Reset);
        }
        
        // Commands
        let cmd = self.parse_command()?;
        Ok(QLangLine::Command(cmd))
    }

    fn parse_command(&mut self) -> Result<QLangCommand, String> {
        println!("Parsing command. Next token: {:?}", self.peek());
        if self.match_token(&[Token::Let]) {
            println!("Matched Let");
            return self.parse_let();
        }
        if self.match_token(&[Token::Fn]) {
            return self.parse_function_def();
        }
        if self.match_token(&[Token::Import]) {
            return self.parse_import();
        }
        if self.match_token(&[Token::If]) {
            return self.parse_if();
        }
        if self.match_token(&[Token::QIf]) {
            return self.parse_qif();
        }
        if self.match_token(&[Token::While]) {
            return self.parse_while();
        }
        if self.match_token(&[Token::For]) {
            return self.parse_for();
        }
        if self.match_token(&[Token::MeasureAll]) {
            // measure_all()
            self.consume(Token::LParen, "Expected '(' after measure_all")?;
            self.consume(Token::RParen, "Expected ')' after measure_all")?;
            return Ok(QLangCommand::MeasureAll);
        }
        if self.match_token(&[Token::Display]) {
            self.consume(Token::LParen, "Expected '(' after display")?;
            self.consume(Token::RParen, "Expected ')' after display")?;
            return Ok(QLangCommand::Display);
        }

        // Identifier start: could be Assign or ApplyGate
        if let Token::Identifier(name) = self.peek().clone() {
            self.advance();
            
            if self.match_token(&[Token::Eq]) {
                // Assignment: name = expr
                let value = self.parse_expression()?;
                return Ok(QLangCommand::Assign { name, value });
            } else if self.match_token(&[Token::LParen]) {
                // Function call: name(args...)
                // Special case: list_functions()
                if name == "list_functions" {
                    self.consume(Token::RParen, "Expected ')' after list_functions")?;
                    return Ok(QLangCommand::ListFunctions);
                }

                // Special case: create(n)
                if name == "create" {
                    let n = self.parse_expression()?;
                    self.consume(Token::RParen, "Expected ')' after create args")?;
                    if let Expression::Number(val) = n {
                         return Ok(QLangCommand::Create(val as usize));
                    } else {
                        return Err("create() expects a number literal".to_string());
                    }
                }
                
                // Special case: measure(args...)
                if name == "measure" {
                    let mut args = Vec::new();
                    if !self.check(Token::RParen) {
                        loop {
                            let expr = self.parse_expression()?;
                            if let Expression::Number(val) = expr {
                                args.push(val as usize);
                            } else {
                                return Err("measure() expects number literals for now".to_string());
                            }
                            if !self.match_token(&[Token::Comma]) { break; }
                        }
                    }
                    self.consume(Token::RParen, "Expected ')' after measure args")?;
                    
                    if args.is_empty() {
                        return Ok(QLangCommand::MeasureAll);
                    } else if args.len() == 1 {
                        return Ok(QLangCommand::Measure(args[0]));
                    } else {
                        return Ok(QLangCommand::MeasureMany(args));
                    }
                }

                // Generic Gate or Function Call
                let mut args = Vec::new();
                if !self.check(Token::RParen) {
                    loop {
                        let expr = self.parse_expression()?;
                        args.push(expr);
                        if !self.match_token(&[Token::Comma]) { break; }
                    }
                }
                self.consume(Token::RParen, "Expected ')' after gate args")?;
                
                // Resolve alias
                let canonical = resolve_alias(&name);
                return Ok(QLangCommand::ApplyGate(canonical.to_string(), args));
            }
        }

        Err(format!("Unexpected token: {:?}", self.peek()))
    }

    fn parse_import(&mut self) -> Result<QLangCommand, String> {
        if let Token::Identifier(path) = self.peek().clone() {
            self.advance();
            Ok(QLangCommand::Import { path })
        } else {
            Err("Expected import path".to_string())
        }
    }

    fn parse_while(&mut self) -> Result<QLangCommand, String> {
        self.consume(Token::LParen, "Expected '(' after 'while'")?;
        let condition = self.parse_expression()?;
        self.consume(Token::RParen, "Expected ')' after condition")?;

        self.consume(Token::LBrace, "Expected '{' before while body")?;
        let mut body = Vec::new();
        while !self.check(Token::RBrace) && !self.is_at_end() {
            body.push(self.parse_command()?);
        }
        self.consume(Token::RBrace, "Expected '}' after while body")?;

        Ok(QLangCommand::While { condition, body })
    }

    fn parse_for(&mut self) -> Result<QLangCommand, String> {
        self.consume(Token::LParen, "Expected '(' after 'for'")?;
        
        let var = if let Token::Identifier(v) = self.peek().clone() {
            self.advance();
            v
        } else {
            return Err("Expected loop variable name".to_string());
        };

        self.consume(Token::In, "Expected 'in' after loop variable")?;

        let start = self.parse_expression()?;
        self.consume(Token::Range, "Expected '..' range operator")?;
        let end = self.parse_expression()?;

        self.consume(Token::RParen, "Expected ')' after for clause")?;

        self.consume(Token::LBrace, "Expected '{' before for body")?;
        let mut body = Vec::new();
        while !self.check(Token::RBrace) && !self.is_at_end() {
            body.push(self.parse_command()?);
        }
        self.consume(Token::RBrace, "Expected '}' after for body")?;

        Ok(QLangCommand::For { var, start, end, body })
    }

    fn parse_function_def(&mut self) -> Result<QLangCommand, String> {
        let name = if let Token::Identifier(n) = self.peek().clone() {
            self.advance();
            n
        } else {
            return Err("Expected function name".to_string());
        };

        self.consume(Token::LParen, "Expected '(' after function name")?;
        let mut params = Vec::new();
        if !self.check(Token::RParen) {
            loop {
                if let Token::Identifier(param) = self.peek().clone() {
                    self.advance();
                    params.push(param);
                } else {
                    return Err("Expected parameter name".to_string());
                }
                if !self.match_token(&[Token::Comma]) { break; }
            }
        }
        self.consume(Token::RParen, "Expected ')' after parameters")?;

        self.consume(Token::LBrace, "Expected '{' before function body")?;
        let mut body = Vec::new();
        while !self.check(Token::RBrace) && !self.is_at_end() {
            body.push(self.parse_command()?);
        }
        self.consume(Token::RBrace, "Expected '}' after function body")?;

        Ok(QLangCommand::FunctionDef { name, params, body })
    }

    fn parse_let(&mut self) -> Result<QLangCommand, String> {
        let name = if let Token::Identifier(n) = self.peek().clone() {
            self.advance();
            n
        } else {
            return Err("Expected variable name after 'let'".to_string());
        };

        self.consume(Token::Eq, "Expected '=' after variable name")?;
        let value = self.parse_expression()?;
        Ok(QLangCommand::Let { name, value })
    }

    fn parse_if(&mut self) -> Result<QLangCommand, String> {
        self.consume(Token::LParen, "Expected '(' after 'if'")?;
        let condition = self.parse_expression()?;
        self.consume(Token::RParen, "Expected ')' after if condition")?;

        self.consume(Token::LBrace, "Expected '{' before if body")?;
        let mut then_branch = Vec::new();
        while !self.check(Token::RBrace) && !self.is_at_end() {
            then_branch.push(self.parse_command()?);
        }
        self.consume(Token::RBrace, "Expected '}' after if body")?;

        let else_branch = if self.match_token(&[Token::Else]) {
            self.consume(Token::LBrace, "Expected '{' before else body")?;
            let mut else_cmds = Vec::new();
            while !self.check(Token::RBrace) && !self.is_at_end() {
                else_cmds.push(self.parse_command()?);
            }
            self.consume(Token::RBrace, "Expected '}' after else body")?;
            Some(else_cmds)
        } else {
            None
        };

        Ok(QLangCommand::If { condition, then_branch, else_branch })
    }

    fn parse_qif(&mut self) -> Result<QLangCommand, String> {
        self.consume(Token::LParen, "Expected '(' after 'qif'")?;
        let condition = self.parse_expression()?;
        self.consume(Token::RParen, "Expected ')' after qif condition")?;

        self.consume(Token::LBrace, "Expected '{' before qif body")?;
        let mut then_branch = Vec::new();
        while !self.check(Token::RBrace) && !self.is_at_end() {
            then_branch.push(self.parse_command()?);
        }
        self.consume(Token::RBrace, "Expected '}' after qif body")?;

        let else_branch = if self.match_token(&[Token::Else]) {
            self.consume(Token::LBrace, "Expected '{' before else body")?;
            let mut else_cmds = Vec::new();
            while !self.check(Token::RBrace) && !self.is_at_end() {
                else_cmds.push(self.parse_command()?);
            }
            self.consume(Token::RBrace, "Expected '}' after else body")?;
            Some(else_cmds)
        } else {
            None
        };

        Ok(QLangCommand::QIf { condition, then_branch, else_branch })
    }

    fn parse_expression(&mut self) -> Result<Expression, String> {
        println!("parse_expression called. Next: {:?}", self.peek());
        self.parse_logic_or()
    }

    fn parse_logic_or(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_logic_and()?;
        while self.match_token(&[Token::Or]) {
            let right = self.parse_logic_and()?;
            left = Expression::BinaryOp { left: Box::new(left), op: Operator::Or, right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_logic_and(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_equality()?;
        while self.match_token(&[Token::And]) {
            let right = self.parse_equality()?;
            left = Expression::BinaryOp { left: Box::new(left), op: Operator::And, right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_comparison()?;
        while self.match_token(&[Token::EqEq, Token::BangEq]) {
            let op = match self.previous() {
                Token::EqEq => Operator::Eq,
                Token::BangEq => Operator::Neq,
                _ => unreachable!(),
            };
            let right = self.parse_comparison()?;
            left = Expression::BinaryOp { left: Box::new(left), op, right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_term()?;
        while self.match_token(&[Token::Lt, Token::Gt, Token::LtEq, Token::GtEq]) {
            let op = match self.previous() {
                Token::Lt => Operator::Lt,
                Token::Gt => Operator::Gt,
                Token::LtEq => Operator::Le,
                Token::GtEq => Operator::Ge,
                _ => unreachable!(),
            };
            let right = self.parse_term()?;
            left = Expression::BinaryOp { left: Box::new(left), op, right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_factor()?;
        while self.match_token(&[Token::Plus, Token::Minus]) {
            let op = match self.previous() {
                Token::Plus => Operator::Add,
                Token::Minus => Operator::Sub,
                _ => unreachable!(),
            };
            let right = self.parse_factor()?;
            left = Expression::BinaryOp { left: Box::new(left), op, right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_unary()?;
        while self.match_token(&[Token::Star, Token::Slash]) {
            let op = match self.previous() {
                Token::Star => Operator::Mul,
                Token::Slash => Operator::Div,
                _ => unreachable!(),
            };
            let right = self.parse_unary()?;
            left = Expression::BinaryOp { left: Box::new(left), op, right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, String> {
        if self.match_token(&[Token::Minus]) {
            let right = self.parse_unary()?;
            // Represent -x as 0 - x
            return Ok(Expression::BinaryOp {
                left: Box::new(Expression::Number(0.0)),
                op: Operator::Sub,
                right: Box::new(right),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expression, String> {
        let mut expr = if self.match_token(&[Token::Number(0.0)]) {
            if let Token::Number(n) = self.previous().clone() {
                Expression::Number(n)
            } else {
                unreachable!() // Should not happen if match_token correctly identifies Number
            }
        } else if self.match_token(&[Token::Identifier("".to_string())]) {
            if let Token::Identifier(name) = self.previous().clone() {
                if name == "measure" {
                    self.consume(Token::LParen, "Expected '(' after measure")?;
                    let expr = self.parse_expression()?;
                    self.consume(Token::RParen, "Expected ')' after measure arg")?;
                    Expression::Measure(Box::new(expr))
                } else {
                    Expression::Variable(name)
                }
            } else {
                unreachable!() // Should not happen if match_token correctly identifies Identifier
            }
        } else if self.match_token(&[Token::LParen]) {
            let expr = self.parse_expression()?;
            self.consume(Token::RParen, "Expected ')' after expression")?;
            expr
        } else if self.match_token(&[Token::LBracket]) {
            // Array literal: [e1, e2, ...]
            let mut exprs = Vec::new();
            if !self.check(Token::RBracket) {
                loop {
                    let expr = self.parse_expression()?;
                    exprs.push(expr);
                    if !self.match_token(&[Token::Comma]) { 
                        break; 
                    }
                }
            }
            self.consume(Token::RBracket, "Expected ']' after array elements")?;
            Expression::Array(exprs)
        } else {
            return Err(format!("Unexpected token in expression: {:?}", self.peek()));
        };

        // Check for indexing: expr[index]
        while self.match_token(&[Token::LBracket]) {
            let index = self.parse_expression()?;
            self.consume(Token::RBracket, "Expected ']' after index")?;
            expr = Expression::Index(Box::new(expr), Box::new(index));
        }

        Ok(expr)
    }

    // Helpers
    fn match_token(&mut self, types: &[Token]) -> bool {
        for t in types {
            if self.check_discriminant(t) {
                self.advance();
                return true;
            }
        }
        false
    }

    fn check(&self, token: Token) -> bool {
        if self.is_at_end() { return false; }
        // Simple equality check might fail for Identifier(String) vs Identifier(String) if content differs
        // But we usually check discriminants or specific tokens like RParen
        std::mem::discriminant(self.peek()) == std::mem::discriminant(&token)
    }

    fn check_discriminant(&self, token: &Token) -> bool {
        if self.is_at_end() { return false; }
        std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
    }

    fn consume(&mut self, token: Token, message: &str) -> Result<Token, String> {
        if self.check_discriminant(&token) {
            Ok(self.advance().clone())
        } else {
            Err(message.to_string())
        }
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_basic() {
        let input = "create(3) x(0)";
        let lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        assert!(matches!(tokens[0], Token::Identifier(_))); // create
        assert!(matches!(tokens[1], Token::LParen));
        assert!(matches!(tokens[2], Token::Number(_)));
        assert!(matches!(tokens[3], Token::RParen));
    }

    #[test]
    fn test_parser_let() {
        let mut parser = QLangParser::new();
        parser.append("let a = 10");
        parser.validate_lines();
        let cmds = parser.get_commands();
        assert!(matches!(cmds[0], QLangLine::Command(QLangCommand::Let { .. })));
    }

    #[test]
    fn test_parser_if() {
        let mut parser = QLangParser::new();
        parser.append("if (a == 1) { x(0) }");
        parser.validate_lines();
        let cmds = parser.get_commands();
        assert!(matches!(cmds[0], QLangLine::Command(QLangCommand::If { .. })));
    }
}
